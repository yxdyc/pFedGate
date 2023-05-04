import copy
import pickle

import torch
from torch.nn.functional import gumbel_softmax
import numpy as np

from learners.learner import Learner
from models.knapsack_solver import KnapsackSolver01, KnapsackSolverFractional
from utils.sparse_factor_schedule import SparsityReduceLROnPlateauScheduler
from utils.torch_utils import get_model_diff_norm

from models.nn_nets import DifferentiableCeilFun, DifferentiableRoundFun

import wandb, logging

DEBUG_WEIGHT_ADAPT = False


class GatedLearner(Learner):
    def __init__(self, base_model, criterion, metric, device, optimizer, lr_scheduler=None,
                 is_binary_classification=False, gating_layer=None, opt_for_gating=None, lr_scheduler_for_gating=None,
                 alpha=0, beta=0., sparse_factor=1., block_wise_prune=False, sparse_factor_scheduler=None,
                 bn_running_stats=True, args_=None):
        super().__init__(base_model, criterion, metric, device, optimizer, lr_scheduler, is_binary_classification)

        self.importance_prior_para_num = args_.importance_prior_para_num
        self.client_level_top_gated_scores = None
        self.gumbel_sigmoid = False
        self.schedule_lr_freq = "epoch"  # ["epoch", "batch"]
        self.alpha = alpha
        self.beta = beta
        self.gating_layer = gating_layer
        self.opt_for_gating = opt_for_gating
        self.person_input_norm = args_.person_input_norm
        if self.person_input_norm == 1:
            self.gating_layer.norm_input_each_forward = False  # the input to gating layer will be normalized
        self.lr_scheduler_for_gating = lr_scheduler_for_gating
        self.sparse_factor = sparse_factor
        self.sparse_factor_scheduler = sparse_factor_scheduler
        if self.sparse_factor_scheduler is not None:
            self.sparse_factor = self.sparse_factor_scheduler.s_begin
        if hasattr(self.model, "set_adapted_forward_mode"):
            self.model.set_adapted_forward_mode()
        if bn_running_stats == 0:
            for module in self.model.modules():
                if hasattr(module, "track_running_stats"):
                    module.track_running_stats = False
        self.min_sparse_factor = min(max(1 / gating_layer.fine_grained_block_split, 0.1), self.sparse_factor)
        self.block_wise_prune = block_wise_prune

        self.feed_batch_count = 0
        self.node_id = -1
        self.global_epoch = 0

        self.total_model_size = torch.sum(self.gating_layer.block_size_lookup_table)
        # mapping block importance predicted by gating layer from continuous [0,1] to integers [importance_scale_factor]
        # such that we can rank the scaled, integer importance scores via 0-1 knapsack solver
        self.gated_scores_scale_factor = 10
        if self.block_wise_prune:
            self.knapsack_solver = KnapsackSolver01(
                value_sum_max=self.gated_scores_scale_factor * len(self.gating_layer.block_size_lookup_table),
                item_num_max=len(self.gating_layer.block_size_lookup_table),
                weight_max=round(self.sparse_factor * self.total_model_size.item())
            )
        else:
            self.knapsack_solver = KnapsackSolverFractional(
                item_num_max=len(self.gating_layer.block_size_lookup_table),
                weight_max=round(self.sparse_factor * self.total_model_size.item())
            )

        self.side_info = dict()  # information sent to & received from server or other clients;
        self.side_info["gating_layer"] = self.gating_layer.gating
        self.side_info["global_metric"] = 0

    def del_adapted_model_para(self):
        # the para within self.model can be deleted to reduce storage cost
        # the real para can be calculated on the fly based on self.meta_model_para

        self.model.del_adapted_para()

    def get_top_gated_scores(self, x):
        """ Get gating weights via the learned gating layer data-dependently """
        # get gating weights data-dependently via gumbel trick
        gating_logits, trans_weights = self.gating_layer(x)  # -> [Batch_size, Num_blocks]
        if self.gumbel_sigmoid:
            # gumbel-sigmoid as softmax of two logits a and 0:  e^a / (e^a + e^0) = 1 / (1 + e^(0 - a)) = sigmoid(a)
            ori_logits_shape = gating_logits.size()
            gating_logits = torch.stack([torch.zeros(ori_logits_shape, device=gating_logits.device),
                                         gating_logits], dim=2)  # -> [Batch_size, Num_blocks, 2]
            gated_scores = gumbel_softmax(gating_logits, hard=False, dim=2)
            gated_scores = gated_scores * torch.stack(
                [torch.zeros(ori_logits_shape, device=gating_logits.device),
                 torch.ones(ori_logits_shape, device=gating_logits.device)], dim=2)
            gated_scores = torch.sum(gated_scores, dim=2)  # -> [Batch_size, Num_blocks]
        else:
            # normed importance score
            gated_scores = torch.sigmoid(gating_logits)
        gated_scores = torch.mean(gated_scores, dim=0)  # -> [Num_blocks]

        # separate trans
        if id(gated_scores) != id(trans_weights):
            # bounded model diff
            trans_weights = torch.sigmoid(trans_weights)
            trans_weights = torch.mean(trans_weights, dim=0)  # -> [Num_blocks]

        # avoid cutting info flow (some internal sub-blocks are all zeros)
        gated_scores = torch.clip(gated_scores, min=self.min_sparse_factor)  # -> [Num_blocks]

        if self.block_wise_prune:
            top_trans_weights, sparse_ratio_selected = self.select_top_trans_weights(gated_scores, trans_weights)
        else:
            top_trans_weights, sparse_ratio_selected = self.select_top_trans_weights(gated_scores, trans_weights,
                                                                                     in_place=False)

        return gated_scores, top_trans_weights, sparse_ratio_selected

        # # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
        # if self.sparse_factor != 1.0:
        #     # self.model.adapted_model_para will be modified in the prune_model function
        #     return gated_scores
        # else:
        #     for para_idx, para in enumerate(self.model.parameters()):
        #         para_name = self.gating_layer.block_names[para_idx]
        #         if not DEBUG_WEIGT_ADAPT:
        #             self.model.adapted_model_para[para_name] = gated_scores[para_idx] * para
        #         else:
        #             self.model.adapted_model_para[para_name] = para * 1  # to DEBUG the adaption

        #     return gated_scores

    def adapt_prune_model(self, top_trans_weights):

        if self.block_wise_prune:
            # get pruned models via with ranked block-wise gating weights
            if self.gating_layer.fine_grained_block_split == 1:
                for para_idx, para in enumerate(self.model.parameters()):
                    mask = torch.ones_like(para, device=self.device).reshape(-1) * top_trans_weights[para_idx]
                    para_name = self.gating_layer.block_names[para_idx]
                    # self.model.adapted_model_para[para_name] = mask * para
                    mask = mask.view(para.shape)
                    self.model.set_adapted_para(para_name, mask * para)
            else:
                for para_name, para in self.model.named_parameters():
                    mask = torch.ones_like(para, device=self.device).reshape(-1)
                    sub_block_begin, sub_block_end, size_each_sub = self.gating_layer.para_name_to_block_split_info[
                        para_name]
                    for i in range(sub_block_begin, sub_block_end):
                        gating_weight_sub_block_i = top_trans_weights[i]
                        block_element_begin = (i - sub_block_begin) * size_each_sub
                        block_element_end = (i + 1 - sub_block_begin) * size_each_sub
                        mask[block_element_begin:block_element_end] *= gating_weight_sub_block_i
                    mask = mask.view(para.shape)
                    # self.model.adapted_model_para[para_name] = mask * para
                    self.model.set_adapted_para(para_name, mask * para)
        else:
            ceil_fun, round_fun = DifferentiableCeilFun.apply, DifferentiableRoundFun.apply
            if self.gating_layer.fine_grained_block_split == 1:
                for para_idx, para in enumerate(self.model.parameters()):
                    ori_size, total_para_num = para.shape, para.numel()
                    mask = torch.ones_like(para, device=self.device).reshape(-1) * top_trans_weights[para_idx]
                    # mask = ceil_fun(mask)  # ones Tensor while keep the grad
                    # select at most the first gated_scores[para_idx] parameters as each dim
                    mask[round_fun(total_para_num * top_trans_weights[para_idx]):] = 0
                    mask = mask.view(ori_size)
                    para_name = self.gating_layer.block_names[para_idx]
                    # self.model.adapted_model_para[para_name] = mask * para
                    self.model.set_adapted_para(para_name, mask * para)
            else:
                for para_name, para in self.model.named_parameters():
                    mask = torch.ones_like(para, device=self.device).reshape(-1)
                    sub_block_begin, sub_block_end, size_each_sub = self.gating_layer.para_name_to_block_split_info[
                        para_name]
                    for i in range(sub_block_begin, sub_block_end):
                        block_element_begin = (i - sub_block_begin) * size_each_sub
                        # select the first gating_weight_sub_block_i para
                        block_element_end_selected = round_fun(
                            (i + 1 - sub_block_begin) * size_each_sub * top_trans_weights[i])
                        block_element_end = (i + 1 - sub_block_begin) * size_each_sub
                        mask[block_element_begin:block_element_end_selected] *= top_trans_weights[i]
                        mask[block_element_end_selected:block_element_end] = 0
                    mask = mask.view(para.shape)
                    # self.model.adapted_model_para[para_name] = mask * para
                    self.model.set_adapted_para(para_name, mask * para)

        return top_trans_weights.detach()

    def select_top_trans_weights(self, gated_scores, trans_weight, in_place=True):
        """
        Keep to sefl.sparse_factor elements of gating weights
        :param gated_scores:
        :param in_place:
        :return:
        """
        if self.sparse_factor == 1:
            return trans_weight, torch.tensor(1.0)
        if in_place:
            retained_trans_weights = trans_weight
        else:
            retained_trans_weights = trans_weight.clone()

        if self.block_wise_prune:
            # keep top (self.sparse_factor) weights via 0-1 knapsack
            mask = torch.ones_like(gated_scores, device=self.device)
            if id(trans_weight) != id(gated_scores):
                # ST trick
                mask = mask - gated_scores.detach() + gated_scores
            if self.importance_prior_para_num == 1:
                importance_value_list = np.array(
                    ((gated_scores + self.gating_layer.block_size_lookup_table_normalized) / 2).tolist())
            else:
                importance_value_list = np.array(gated_scores.tolist())
            importance_value_list = np.around(importance_value_list * self.gated_scores_scale_factor).astype(np.int)

            # for linear_layer sub_blocks
            linear_layer_block_idx_filter_first = self.gating_layer.linear_layer_block_idx_filter_first
            selected_size = self._select_top_sub_blocks(importance_value_list, linear_layer_block_idx_filter_first,
                                                        mask)

            # for non-linear-layer sub_blocks
            non_linear_layer_block_idx_filter_first = self.gating_layer.non_linear_layer_block_idx_filter_first
            selected_size += self._select_top_sub_blocks(importance_value_list, non_linear_layer_block_idx_filter_first,
                                                         mask)

            retained_trans_weights *= mask
        else:
            trans_weights_after_select = torch.full_like(trans_weight, fill_value=self.min_sparse_factor)

            # for linear_layer sub_blocks
            linear_layer_block_idx = self.gating_layer.linear_layer_block_idx
            selected_size = self._select_top_sub_blocks_frac(gated_scores, linear_layer_block_idx,
                                                             trans_weights_after_select)

            # for non-linear-layer sub_blocks
            non_linear_layer_block_idx = self.gating_layer.non_linear_layer_block_idx
            selected_size += self._select_top_sub_blocks_frac(retained_trans_weights, non_linear_layer_block_idx,
                                                              trans_weights_after_select)

            calibration_quantity = trans_weights_after_select - trans_weight.detach()
            retained_trans_weights += calibration_quantity

        return retained_trans_weights, selected_size / self.total_model_size

    def _select_top_sub_blocks(self, importance_value_list, block_idx, mask):
        weight_list = self.gating_layer.block_size_lookup_table[block_idx]
        importance_value_list = importance_value_list[block_idx]
        capacity = torch.round(torch.sum(weight_list) * (self.sparse_factor - self.min_sparse_factor)).int()
        total_value_of_selected_items, total_weight, selected_item_idx, droped_item_idx = self.knapsack_solver.found_max_value_greedy(
            weight_list=weight_list.tolist(),
            value_list=importance_value_list,
            capacity=capacity
        )
        # droped_item_idx = [i for i in range(len(block_idx)) if
        #                    i not in selected_item_idx[0].tolist()]
        droped_item_idx = np.array(block_idx)[droped_item_idx]
        mask[droped_item_idx] *= 0

        if isinstance(total_weight, torch.Tensor):
            # return sum(weight_list[selected_item_idx]).detach()
            return total_weight.detach()
        else:
            return total_weight

    def _select_top_sub_blocks_frac(self, importance_value_list, block_idx, gated_scores_after_select):
        # to make the minimal gating weights of each block as self.min_sparse_factor,
        # we allocate the remaining capacity (self.sparse_model_size - self.min_sparse_factor)
        # into blocks according to their importance value (gating weights)
        weight_list = self.gating_layer.block_size_lookup_table[block_idx]
        importance_value_list = importance_value_list[block_idx]
        capacity = torch.sum(weight_list) * (self.sparse_factor - self.min_sparse_factor)
        total_value_of_selected_items, selected_items_weight, selected_items_frac = self.knapsack_solver.found_max_value(
            weight_list=weight_list * (1 - self.min_sparse_factor),
            value_list=importance_value_list,
            capacity=capacity
        )
        # to make the backward work, we add a calibration tensor onto gating weights,
        # such that the gated_scores close to the results from knapsack_solver
        gated_scores_after_select[block_idx] += selected_items_weight / weight_list

        return sum(selected_items_weight).detach()

    def fit_batch(self, batch,
                  train_gating_layer=False, train_base_model=True, sample_level_adapt=True):
        """
        perform an optimizer step over one batch drawn from `iterator`
        Bi-Level optimization for the gating layer and model_to_mask

        :param batch: tuple of (x, y, indices)
        :param train_gating_layer: whether train paras of gating layer
        :param train_base_model: whether train paras of base model
        :return:
            loss.detach()
            metric.detach()

        """
        if train_gating_layer:
            self.gating_layer.train()
            self.opt_for_gating.zero_grad()
        else:
            self.gating_layer.eval()
        if train_base_model:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        x, y, indices = batch
        x = x.to(self.device).type(torch.float32)
        y = y.to(self.device)

        if self.person_input_norm:
            x = self.gating_layer.norm_input_layer(x)

        if self.is_binary_classification:
            y = y.type(torch.float32).unsqueeze(1)

        # the adapted model varies for different samples
        if sample_level_adapt:
            # get personalized masks via gating layer
            gated_scores, top_trans_weights, sparse_ratio_selected = self.get_top_gated_scores(x)
            # mask the meta-model according to sparsity preference
            top_trans_weights = self.adapt_prune_model(top_trans_weights)
        else:
            top_trans_weights = self.client_level_top_gated_scores
            sparse_ratio_selected = self.min_sparse_factor

        y_pred = self.model.adapted_forward(x)
        loss_vec = self.criterion(y_pred, y)
        metric = self.metric(y_pred, y) / len(y)
        loss_meta_model = loss_vec.mean()

        loss_diversity = get_model_diff_norm(dict(list(self.model.named_parameters())),
                                             self.model.adapted_model_para)
        # In the final version, the loss_diversity term is not used as we always set alpha=0
        loss_gating_layer = loss_meta_model + self.alpha / 2 * loss_diversity

        if train_gating_layer:
            loss_gating_layer.backward()
            torch.nn.utils.clip_grad_norm_(self.gating_layer.parameters(), max_norm=10, norm_type=2)
            self.opt_for_gating.step()
            if self.lr_scheduler_for_gating and self.schedule_lr_freq == "batch":
                self.lr_scheduler_for_gating.step()
            self.side_info["gating_layer"] = self.gating_layer.gating  # transfer weights of gating layer to server

        if train_base_model:
            if not train_gating_layer:
                if sample_level_adapt:
                    loss_meta_model.backward()
                else:
                    with torch.autograd.set_detect_anomaly(True):
                        loss_meta_model.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10, norm_type=2)
            self.optimizer.step()
            if self.lr_scheduler and self.schedule_lr_freq == "batch":
                self.lr_scheduler.step()

        if sample_level_adapt:
            # the adapted model varies for different samples
            self.del_adapted_model_para()

        return loss_meta_model.detach(), metric.detach(), sparse_ratio_selected.detach(), top_trans_weights.detach(), \
               loss_diversity.detach() if isinstance(loss_diversity, torch.Tensor) else loss_diversity

    def fit_epoch(self, iterator, train_gating_layer=False, train_base_model=True, sample_level_adapt=True):
        total_loss_meta_model = 0
        total_loss_diversity = 0
        total_metric = 0
        total_top_gated_scores = torch.zeros(len(self.gating_layer.block_size_lookup_table), device=self.device)
        total_sparse_ratio_selected = 0

        n_samples = 0

        for x, y, indices in iterator:
            x = x.to(self.device)
            y = y.to(self.device)
            batch_samples = y.size(0)
            if batch_samples == 1:
                continue  # for batch_norm. Otherwise we got `Error: Expected more than 1 value per channel when training`
            n_samples += batch_samples
            loss_meta_model, metric, sparse_ratio_selected, top_gated_scores, loss_diversity = self.fit_batch(
                (x, y, indices),
                train_gating_layer,
                train_base_model,
                sample_level_adapt)
            self.feed_batch_count += 1
            total_loss_meta_model += loss_meta_model * batch_samples
            total_loss_diversity += loss_diversity * batch_samples
            total_metric += metric * batch_samples
            total_sparse_ratio_selected += sparse_ratio_selected * batch_samples
            total_top_gated_scores += top_gated_scores * batch_samples

            try:
                # commit=False makes the global step of wandb not incremented, such that we can log for different nodes
                wandb.log({
                    f"node{self.node_id}/main/loss_meta_model": loss_meta_model,
                    f"node{self.node_id}/main/loss_diversity": loss_diversity,
                    f"node{self.node_id}/main/sparse_ratio_selected": sparse_ratio_selected,
                    f"node{self.node_id}/main/metric": metric,
                    f"node{self.node_id}/main/feed_batch_count": self.feed_batch_count,
                    f"node{self.node_id}/main/sparse_factor": self.sparse_factor,
                    f"node{self.node_id}/main/lr_base_model": self.optimizer.param_groups[0]['lr'],
                    f"node{self.node_id}/main/lr_gating": self.opt_for_gating.param_groups[0]['lr']
                },
                    commit=False)
            except ModuleNotFoundError:
                logging.warning("not found wandb, will not track wandb related results")
        self.global_epoch += 1
        if train_gating_layer and self.lr_scheduler_for_gating and self.schedule_lr_freq == "epoch":
            if isinstance(self.lr_scheduler_for_gating, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler_for_gating.step(self.side_info["global_metric"], epoch=self.global_epoch)
            else:
                self.lr_scheduler_for_gating.step(epoch=self.global_epoch, )
            if self.sparse_factor_scheduler:
                if isinstance(self.sparse_factor_scheduler, SparsityReduceLROnPlateauScheduler):
                    self.sparse_factor = self.sparse_factor_scheduler.step(self.side_info["global_metric"],
                                                                           epoch=self.global_epoch)
                else:
                    self.sparse_factor = self.sparse_factor_scheduler.step(
                                         epoch=self.global_epoch)

        if train_base_model and self.lr_scheduler and self.schedule_lr_freq == "epoch":
            if isinstance(self.lr_scheduler_for_gating, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(self.side_info["global_metric"], epoch=self.global_epoch)
            else:
                self.lr_scheduler.step(epoch=self.global_epoch)

        total_loss_meta_model /= n_samples
        total_loss_diversity /= n_samples
        total_metric /= n_samples
        total_sparse_ratio_selected /= n_samples
        total_top_gated_scores /= n_samples

        return total_loss_meta_model, total_metric, total_sparse_ratio_selected, total_top_gated_scores, n_samples

    def fit_epochs(self, iterator, n_epochs, weights=None, sample_level_adapt=True):
        for step in range(n_epochs):
            self.fit_epoch(iterator, weights, sample_level_adapt)

    def client_level_adapt(self, iterator, train_gating_layer=False, train_base_model=False):
        if train_gating_layer:
            self.gating_layer.train()
            self.opt_for_gating.zero_grad()
        else:
            self.gating_layer.eval()
        if train_base_model:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()
        batch_num = 0
        gated_scores = 0
        top_gated_scores = 0
        # get statistics from the whole iterator
        for x, y, indices in iterator:
            x = x.to(self.device).type(torch.float32)
            # get personalized masks via gating layer
            gated_score, top_gated_score, sparse_ratio_selected = self.get_top_gated_scores(x)
            gated_scores += gated_score
            top_gated_scores += top_gated_score
            batch_num += 1
        gated_scores /= batch_num
        top_gated_scores /= batch_num
        # mask the meta-model according to sparsity preference
        top_gated_scores = self.adapt_prune_model(top_gated_scores)
        self.client_level_top_gated_scores = top_gated_scores
