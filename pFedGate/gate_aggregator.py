import copy
import logging
import os

import numpy as np
import torch
import wandb

from aggregator import Aggregator
from utils.torch_utils import average_model_of_learners, average_torch_modules, average_torch_state_dict_online, \
    mean_torch_state_dict, \
    copy_side_info, copy_model


class pFedGateAggregator(Aggregator):
    r""" personalized FedGate Aggregator
    Compared to the CentralizedAggregator,
    1. need to average model weights, as well as the client-specific gating weights
    2. need to recover sparse models according to idx

    """

    def __init__(
            self,
            clients,
            global_learners_ensemble,
            log_freq,
            global_train_logger,
            global_test_logger,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            verbose=0,
            seed=None,
            aggregate_sampled_clients=0,
            online_aggregate=0,
            outdir=".",
            *args,
            **kwargs
    ):
        self.best_test_acc = 0.0
        self.aggregate_sampled_clients = aggregate_sampled_clients

        super().__init__(clients, global_learners_ensemble, log_freq, global_train_logger, global_test_logger,
                         sampling_rate, sample_with_replacement, test_clients, verbose, seed, outdir, args, kwargs)
        self.online_aggregate = online_aggregate
        self.trans_gating_layer = False
        self.global_model = self.global_learners_ensemble[0].model
        self.clients_id_2_weights = {client.node_id: client.n_train_samples / self.total_clients_train_num for i, client
                                     in enumerate(self.clients)}
        self.online_aggregated_model = None

        # used in online logging mode
        self.global_val_loss = 0.
        self.global_val_sparse_ratio_selected = 0.
        self.global_val_acc = 0.
        self.global_test_loss = 0.
        self.global_test_sparse_ratio_selected = 0.
        self.global_test_acc = 0.
        block_num = len(self.clients[0].learners_ensemble.learners[0].gating_layer.block_size_lookup_table)
        self.global_val_top_gating_weights = torch.zeros(block_num, device=self.device)
        self.global_test_top_gating_weights = torch.zeros(block_num, device=self.device)

        self.test_acc_all_clients = []
        from utils.utils import RunningStats
        self.running_stats_for_gating_weight = RunningStats()

        self.total_n_val_samples = 0
        self.total_n_test_samples = 0

    def mix(self):
        self.sample_clients()
        if self.online_aggregate == 1:
            self._mix_load_three_model()
        else:
            self._mix_load_all_model()

    def _mix_load_all_model(self):
        sampled_client_trained_n = []
        # Training
        for client in self.sampled_clients:
            # lazy update: assign the global model and global side-info the sampled clients
            self.global_to_local_info(client)
            # client-step: local actions: training and sending (in standalone simulation, no sending)
            total_loss_meta_model, total_metric, total_sparse_ratio_selected, total_top_gating_weights, trained_sample_n = client.step()
            sampled_client_trained_n.append(trained_sample_n)

        # Aggregate
        for learner_id, learner in enumerate(self.global_learners_ensemble):
            if self.aggregate_sampled_clients == 1:
                agg_weights = torch.Tensor(sampled_client_trained_n, device=self.clients_weights.device)
                agg_weights = agg_weights / agg_weights.sum()
                learners = [client.learners_ensemble[learner_id] for client in self.sampled_clients]
                average_model_of_learners(learners, learner, weights=agg_weights)
                ori_gating_weights = [client.learners_ensemble[learner_id].side_info["gating_layer"] for client in
                                      self.sampled_clients]
                if self.trans_gating_layer:
                    average_torch_modules(ori_gating_weights, learner.side_info["gating_layer"], weights=agg_weights)
            else:
                learners = [client.learners_ensemble[learner_id] for client in self.clients]
                average_model_of_learners(learners, learner, weights=self.clients_weights)
                ori_gating_weights = [client.learners_ensemble[learner_id].side_info["gating_layer"] for client in
                                      self.clients]
                if self.trans_gating_layer:
                    average_torch_modules(ori_gating_weights, learner.side_info["gating_layer"],
                                          weights=self.clients_weights)
        self.c_round += 1

        # Evaluate
        if self.c_round % self.log_freq == 0:
            # to eval all clients, we need assign the updated model to the non-sampled clients
            non_sampled_clients = [client for client in self.clients if client not in self.sampled_clients]
            self.update_clients(update_set=non_sampled_clients)
            # evaluate over all clients
            self.write_logs()

    def _mix_load_three_model(self):
        """
            To handle limited GPU memory with large number of clients case,
            we save space by maintaining only three model objects for the FL process
            _shared_local_update_model,  online_aggregated_model,  global_model

            (1) all clients only initialize one model object, and share it (_shared_local_update_model)
            (2) in each training round, the _shared_local_update_model will load_state_dict from global_model, then update online_aggregated_model with trained _shared_local_update_model;
            (3) at the end of training round, global_model will load_state_dict from online_aggregated_model
        :return:
        """
        if self.trans_gating_layer:
            raise NotImplementedError("Not implement transfer gating layer")
        sampled_client_trained_n = []
        self.online_aggregated_model = copy.deepcopy(self.global_model).state_dict(keep_vars=True)
        for key in self.online_aggregated_model:
            self.online_aggregated_model[key].data = torch.zeros_like(self.online_aggregated_model[key])

        # Training & Online aggregate
        self.c_round += 1
        online_evaluate_all_clients = True if self.c_round % self.log_freq == 0 else False
        if online_evaluate_all_clients:
            self.reset_global_eval_res()

        for i, client in enumerate(self.sampled_clients):
            # lazy broadcast: global_model -> _shared_local_update_model
            self.global_to_local_info(client)

            # client-step: local actions: training and sending (in standalone simulation, no sending)
            total_loss_meta_model, total_metric, total_sparse_ratio_selected, total_top_gating_weights, trained_sample_n = client.step()
            sampled_client_trained_n.append(trained_sample_n)
            if online_evaluate_all_clients:
                logging.debug(f"==>  Evaluating sampled client with id {client.node_id}. | {i}/{len(self.clients)}")
                test_metric, val_metric = self.write_logs_online(client, log_all=False)
                self.cross_eval_clinet(client, self_test_metric=test_metric, self_val_metric=val_metric)

            # update online_aggregated_model via trained local model
            average_torch_state_dict_online(ori_state_dict=client.learners_ensemble[0].model.state_dict(keep_vars=True),
                                            ori_module_weight=trained_sample_n,
                                            target_state_dict=self.online_aggregated_model)

        if self.aggregate_sampled_clients == 1:
            mean_torch_state_dict(self.online_aggregated_model,
                                  total_weights=sum(sampled_client_trained_n))
        else:  # i.e., the weighted sum is over all clients
            num_not_sampled_clients = len(self.clients) - len(self.sampled_clients)
            not_sampled_total_trained_n = 0
            if num_not_sampled_clients > 0:
                non_sampled_clients = [client for client in self.clients if client not in self.sampled_clients]
                for client in non_sampled_clients:
                    not_sampled_total_trained_n += client.n_train_samples
                # non sampled clients still have the same model as self.global_model
                average_torch_state_dict_online(ori_state_dict=self.global_model.state_dict(keep_vars=True),
                                                ori_module_weight=not_sampled_total_trained_n,
                                                target_state_dict=self.online_aggregated_model)
            mean_torch_state_dict(self.online_aggregated_model,
                                  total_weights=not_sampled_total_trained_n + sum(sampled_client_trained_n))

        # at the end of training round, update global_model
        copy_model(target=self.global_model, source=self.online_aggregated_model)
        if online_evaluate_all_clients:
            # evaluate the non-sampled ones, whose model is the same as global_model
            num_not_sampled_clients = len(self.clients) - len(self.sampled_clients)
            if num_not_sampled_clients > 0:
                non_sampled_clients = [client for client in self.clients if client not in self.sampled_clients]
                for i, client in enumerate(non_sampled_clients):
                    logging.debug(
                        f"==>  Evaluating non-sampled client with id {client.node_id}. | {i + len(self.sampled_clients)}/{len(self.clients)}")
                    test_metric, val_metric = self.write_logs_online(client, log_all=False)
            if self.c_round in self.cross_eval_rounds:
                self.cross_eval_clinet(client, self_test_metric=test_metric, self_val_metric=val_metric)
            # summarize the results and log them
            self.write_logs_online(client=None, log_all=True)

    def cross_eval_clinet(self, client, self_test_metric, self_val_metric):
        if self.c_round in self.cross_eval_rounds:
            for target_client in self.clients:
                # eval current client on other target_clients' data
                if target_client == client:
                    self.cross_eval_res_test[int(client.node_id), int(client.node_id)] = self_test_metric
                    self.cross_eval_res_val[int(client.node_id), int(client.node_id)] = self_val_metric
                else:
                    val_loss_meta_model, val_metric, val_sparse_ratio_selected, val_top_gating_weights, \
                    test_loss_meta_model, test_metric, test_sparse_ratio_selected, test_top_gating_weights = \
                        client.write_logs(
                            test_iter=target_client.test_iterator,
                            val_iter=target_client.val_iterator,
                            node_setting=f"cross_eval_res/node{client.node_id}on{target_client.node_id}")
                    self.cross_eval_res_test[int(client.node_id), int(target_client.node_id)] = test_metric
                    self.cross_eval_res_val[int(client.node_id), int(target_client.node_id)] = val_metric

    def update_clients(self, update_set=None):
        if update_set is None:
            update_set = self.clients
        for client in update_set:
            self.global_to_local_info(client)

    def global_to_local_info(self, target_client):
        # assign the updated global model and global side-info to the target clients
        for learner_id, learner in enumerate(target_client.learners_ensemble):
            # learner.set_meta_model_para(self.global_learners_ensemble[learner_id].model.parameters())
            copy_model(target=learner.model, source=self.global_learners_ensemble[learner_id].model)
            if self.trans_gating_layer:
                copy_side_info(target=learner.side_info, source=self.global_learners_ensemble[learner_id].side_info)
            else:
                copy_side_info(target=learner.side_info, source=self.global_learners_ensemble[learner_id].side_info,
                               filter_keys=["gating_layer"])

            if callable(getattr(learner.optimizer, "set_initial_params", None)):
                learner.optimizer.set_initial_params(
                    self.global_learners_ensemble[learner_id].model.parameters()
                )
            learner.global_epoch = self.c_round

    def write_logs(self):
        self.update_test_clients()

        for global_logger, clients in [
            (self.global_train_logger, self.clients),
            (self.global_test_logger, self.test_clients)
        ]:
            if len(clients) == 0:
                continue

            global_val_loss = 0.
            global_val_sparse_ratio_selected = 0.
            global_val_acc = 0.
            global_test_loss = 0.
            global_test_sparse_ratio_selected = 0.
            global_test_acc = 0.
            block_num = len(self.clients[0].learners_ensemble.learners[0].gating_layer.block_size_lookup_table)
            global_val_top_gating_weights = torch.zeros(block_num, device=self.device)
            global_test_top_gating_weights = torch.zeros(block_num, device=self.device)

            test_acc_all_clients = []
            from utils.utils import RunningStats
            running_stats_for_gating_weight = RunningStats()

            total_n_val_samples = 0
            total_n_test_samples = 0

            for i, client in enumerate(clients):
                logging.debug(f"==>  Evaluating client with id {client.node_id}. | {i}/{len(clients)}")
                val_loss_meta_model, val_metric, val_sparse_ratio_selected, val_top_gating_weights, \
                test_loss_meta_model, test_metric, test_sparse_ratio_selected, test_top_gating_weights = client.write_logs()

                global_val_loss += val_loss_meta_model * client.n_val_samples
                global_val_sparse_ratio_selected += val_sparse_ratio_selected
                global_val_acc += val_metric * client.n_val_samples
                global_val_top_gating_weights += val_top_gating_weights * client.n_val_samples

                global_test_loss += test_loss_meta_model * client.n_test_samples
                global_test_sparse_ratio_selected += test_sparse_ratio_selected
                global_test_acc += test_metric * client.n_test_samples
                global_test_top_gating_weights += test_top_gating_weights * client.n_test_samples

                test_acc_all_clients.append(test_metric)

                running_stats_for_gating_weight.push(test_top_gating_weights)
                self.client_level_mask[int(client.node_id)] = test_top_gating_weights

                total_n_val_samples += client.n_val_samples
                total_n_test_samples += client.n_test_samples
                self.cross_eval_clinet(client, self_test_metric=test_metric, self_val_metric=val_metric)

            global_test_loss /= total_n_test_samples
            global_test_acc /= total_n_test_samples
            global_test_sparse_ratio_selected /= len(self.clients)
            global_test_top_gating_weights /= total_n_test_samples
            if total_n_val_samples != 0:
                global_val_loss /= total_n_val_samples
                global_val_acc /= total_n_val_samples
                global_val_sparse_ratio_selected /= len(self.clients)
                global_val_top_gating_weights /= total_n_val_samples

            test_acc_all_clients.sort()
            bottom_decile_acc = test_acc_all_clients[len(test_acc_all_clients) // 10]

            self.global_learners_ensemble[0].side_info["global_metric"] = global_test_acc
            log_res = {f"node_agg/val/loss_meta_model": global_val_loss,
                       f"node_agg/val/sparse_ratio_selected": global_val_sparse_ratio_selected,
                       f"node_agg/val/metric": global_val_acc,
                       f"node_agg/val/c_round": self.c_round,
                       f"node_agg/test/loss_meta_model": global_test_loss,
                       f"node_agg/test/sparse_ratio_selected": global_test_sparse_ratio_selected,
                       f"node_agg/test/metric": global_test_acc,
                       f"node_agg/test/metric_bottom_client": bottom_decile_acc,
                       f"node_agg/test/c_round": self.c_round}
            logging.info(log_res)

            try:
                block_names = self.global_learners_ensemble[0].gating_layer.block_names
                val_bar_data = [[name, weight] for (name, weight) in
                                zip(block_names, global_val_top_gating_weights.tolist())]
                val_table = wandb.Table(data=val_bar_data, columns=["block_name", "gating_weight"])
                test_bar_data = [[name, weight] for (name, weight) in
                                 zip(block_names, global_test_top_gating_weights.tolist())]
                test_table = wandb.Table(data=test_bar_data, columns=["block_name", "gating_weight"])
                test_gating_std_bar_data = [[name, weight] for (name, weight) in
                                            zip(block_names,
                                                running_stats_for_gating_weight.standard_deviation().tolist())]
                test_gating_std_table = wandb.Table(data=test_gating_std_bar_data,
                                                    columns=["block_name", "gating_weight_std"])
                log_res.update({
                    f"node_agg/val/top_gating_weights_hist": wandb.Histogram(global_val_top_gating_weights.tolist()),
                    f"node_agg/test/top_gating_weights_hist": wandb.Histogram(global_test_top_gating_weights.tolist()),
                    f"node_agg/val/top_gating_weights": wandb.plot.bar(val_table, "block_name", "gating_weight",
                                                                       title="Val, Per Block Gating Weight"),
                    f"node_agg/test/top_gating_weights": wandb.plot.bar(test_table, "block_name", "gating_weight",
                                                                        title="Test, Per Block Gating Weight"),
                    f"node_agg/test/top_gating_weights_std": wandb.plot.bar(
                        test_gating_std_table, "block_name", "gating_weight_std",
                        title="Test, Gating Weight Std over All Clients"),
                })

                if global_test_acc > self.best_test_acc:
                    self.best_test_acc = global_test_acc
                    log_res.update({f"node_agg/test/best_metric": self.best_test_acc})
                    # save masks
                    masks = torch.stack(self.client_level_mask).cpu().numpy()
                    np.save(file=os.path.join(self.outdir, f"client_level_masks_round_{self.c_round}"),  #
                            arr=masks)
                    found_files = self._find_files_similar_subwords(subwords="client_level_masks_round",
                                                                    find_dir=self.outdir)
                    # delete stale masks
                    retains_n = 3
                    n_removes = len(found_files) - 3
                    if n_removes > 0:
                        # logging.info("found found_files: {}".format(found_files))
                        for time, filename in found_files[:n_removes]:
                            logging.info(
                                "To retain {} files, will del the stale one {}".format(retains_n, filename))
                            if os.path.exists(filename):
                                os.remove(filename)
                    self.client_level_mask = [None for _ in range(len(self.clients))]

                # commit=False makes the global step of wandb not incremented, such that we can log for different nodes
                wandb.log(log_res, commit=True)
            except ModuleNotFoundError:
                logging.warning("not found wandb, will not track wandb related results")

    def reset_global_eval_res(self):
        self.global_train_loss = 0.
        self.global_train_acc = 0.
        self.global_test_loss = 0.
        self.global_test_acc = 0.

        self.total_n_samples = 0
        self.total_n_test_samples = 0

        self.global_val_loss = 0.
        self.global_val_sparse_ratio_selected = 0.
        self.global_val_acc = 0.
        self.global_test_loss = 0.
        self.global_test_sparse_ratio_selected = 0.
        self.global_test_acc = 0.
        block_num = len(self.clients[0].learners_ensemble.learners[0].gating_layer.block_size_lookup_table)
        self.global_val_top_gating_weights = torch.zeros(block_num, device=self.device)
        self.global_test_top_gating_weights = torch.zeros(block_num, device=self.device)

        self.test_acc_all_clients = []
        self.running_stats_for_gating_weight.reset()

        self.total_n_val_samples = 0
        self.total_n_test_samples = 0

    def write_logs_online(self, client, log_all=False):
        if client:
            val_loss_meta_model, val_metric, val_sparse_ratio_selected, val_top_gating_weights, \
            test_loss_meta_model, test_metric, test_sparse_ratio_selected, test_top_gating_weights = client.write_logs()

            self.global_val_loss += val_loss_meta_model * client.n_val_samples
            self.global_val_sparse_ratio_selected += val_sparse_ratio_selected
            self.global_val_acc += val_metric * client.n_val_samples
            self.global_val_top_gating_weights += val_top_gating_weights * client.n_val_samples

            self.global_test_loss += test_loss_meta_model * client.n_test_samples
            self.global_test_sparse_ratio_selected += test_sparse_ratio_selected
            self.global_test_acc += test_metric * client.n_test_samples
            self.global_test_top_gating_weights += test_top_gating_weights * client.n_test_samples

            self.test_acc_all_clients.append(test_metric)

            self.running_stats_for_gating_weight.push(test_top_gating_weights)

            self.total_n_val_samples += client.n_val_samples
            self.total_n_test_samples += client.n_test_samples
            self.client_level_mask[int(client.node_id)] = test_top_gating_weights

            return test_metric, val_metric

        if log_all:
            self.global_test_loss /= self.total_n_test_samples
            self.global_test_acc /= self.total_n_test_samples
            self.global_test_sparse_ratio_selected /= len(self.clients)
            self.global_test_top_gating_weights /= self.total_n_test_samples
            if self.total_n_val_samples != 0:
                self.global_val_loss /= self.total_n_val_samples
                self.global_val_acc /= self.total_n_val_samples
                self.global_val_sparse_ratio_selected /= len(self.clients)
                self.global_val_top_gating_weights /= self.total_n_val_samples

            self.test_acc_all_clients.sort()
            bottom_decile_acc = self.test_acc_all_clients[len(self.test_acc_all_clients) // 10]

            self.global_learners_ensemble[0].side_info["global_metric"] = self.global_test_acc
            log_res = {f"node_agg/val/loss_meta_model": self.global_val_loss,
                       f"node_agg/val/sparse_ratio_selected": self.global_val_sparse_ratio_selected,
                       f"node_agg/val/metric": self.global_val_acc,
                       f"node_agg/val/c_round": self.c_round,
                       f"node_agg/test/loss_meta_model": self.global_test_loss,
                       f"node_agg/test/sparse_ratio_selected": self.global_test_sparse_ratio_selected,
                       f"node_agg/test/metric": self.global_test_acc,
                       f"node_agg/test/metric_bottom_client": bottom_decile_acc,
                       f"node_agg/test/c_round": self.c_round}
            logging.info(log_res)

            # Cross-Evaluation Results
            if self.c_round in self.cross_eval_rounds:
                # save the whole matrix locally
                self.save_cross_eval_res(log_res)
            try:
                block_names = self.global_learners_ensemble[0].gating_layer.block_names
                val_bar_data = [[name, weight] for (name, weight) in
                                zip(block_names, self.global_val_top_gating_weights.tolist())]
                val_table = wandb.Table(data=val_bar_data, columns=["block_name", "gating_weight"])
                test_bar_data = [[name, weight] for (name, weight) in
                                 zip(block_names, self.global_test_top_gating_weights.tolist())]
                test_table = wandb.Table(data=test_bar_data, columns=["block_name", "gating_weight"])
                # test_gating_std_bar_data = [[name, weight] for (name, weight) in
                #                             zip(block_names,
                #                                 self.running_stats_for_gating_weight.standard_deviation().tolist())]
                # test_gating_std_table = wandb.Table(data=test_gating_std_bar_data,
                #                                     columns=["block_name", "gating_weight"])
                log_res.update({
                    f"node_agg/val/top_gating_weights_hist": wandb.Histogram(
                        self.global_val_top_gating_weights.tolist()),
                    f"node_agg/test/top_gating_weights_hist": wandb.Histogram(
                        self.global_test_top_gating_weights.tolist()),
                    f"node_agg/val/top_gating_weights": wandb.plot.bar(val_table, "block_name", "gating    _weight",
                                                                       title="Val, Per Block Gating Weight"),
                    f"node_agg/test/top_gating_weights": wandb.plot.bar(test_table, "block_name", "gating    _weight",
                                                                        title="Test, Per Block Gating     Weight"),
                    # f"node_agg/test/top_gating_weights_std": wandb.plot.bar(
                    #     test_gating_std_table, "block_name", "gating_weight_std",
                    #     title="Test, Gating Weight Std over All Clients"),
                })

                if self.global_test_acc > self.best_test_acc:
                    self.best_test_acc = self.global_test_acc
                    log_res.update({f"node_agg/test/best_metric": self.best_test_acc})

                    # save masks
                    masks = torch.stack(self.client_level_mask).cpu().numpy()
                    np.save(file=os.path.join(self.outdir, f"client_level_masks_round_{self.c_round}"),  #
                            arr=masks)
                    found_files = self._find_files_similar_subwords(subwords="client_level_masks_round",
                                                                    find_dir=self.outdir)
                    # delete stale masks
                    retains_n = 3
                    n_removes = len(found_files) - 3
                    if n_removes > 0:
                        # logging.info("found found_files: {}".format(found_files))
                        for time, filename in found_files[:n_removes]:
                            logging.info(
                                "To retain {} files, will del the stale one {}".format(retains_n, filename))
                            if os.path.exists(filename):
                                os.remove(filename)
                    self.client_level_mask = [None for _ in range(len(self.clients))]
                wandb.log(log_res, commit=True)
            except ModuleNotFoundError:
                logging.warning("not found wandb, will not track wandb related results")

            return None, None

    def _find_files_similar_subwords(self, subwords, find_dir):
        matched_files = (file for file in os.listdir(find_dir) if subwords in file)

        def _prepend_mtime(f, dir_name):
            t = os.stat(os.path.join(dir_name, f)).st_mtime
            return (t, f)

        return sorted(_prepend_mtime(file, find_dir) for file in matched_files)

    def save_cross_eval_res(self, log_res):
        np.save(file=os.path.join(self.outdir, f"cross_eval_res_round_{self.c_round}"),
                arr=self.cross_eval_res_test)
        client_labels = np.array([f"Client {i}" for i in range(len(self.clients))])
        all_results = self.cross_eval_res_test.diagonal()
        all_sorted_idx = np.argsort(-all_results)
        top_30_results_idx = all_sorted_idx[:30]
        top_30_results_idx.sort()
        top_30_results_matrix = self.cross_eval_res_test[top_30_results_idx, :][:, top_30_results_idx]
        log_res.update({f"node_agg/cross_eval_res/round_{self.c_round}_top30_clients":
                            wandb.plots.HeatMap(x_labels=client_labels[top_30_results_idx],
                                                y_labels=client_labels[top_30_results_idx],
                                                matrix_values=top_30_results_matrix)})
        top_50_results_idx = all_sorted_idx[:50]
        top_50_results_idx.sort()
        top_50_results_matrix = self.cross_eval_res_test[top_50_results_idx, :][:, top_50_results_idx]
        log_res.update({f"node_agg/cross_eval_res/round_{self.c_round}_top50_clients":
                            wandb.plots.HeatMap(x_labels=client_labels[top_50_results_idx],
                                                y_labels=client_labels[top_50_results_idx],
                                                matrix_values=top_50_results_matrix)})
        top_100_results_idx = all_sorted_idx[:100]
        top_100_results_idx.sort()
        top_100_results_matrix = self.cross_eval_res_test[top_100_results_idx, :][:, top_100_results_idx]
        log_res.update({f"node_agg/cross_eval_res/round_{self.c_round}_top100_clients":
                            wandb.plots.HeatMap(x_labels=client_labels[top_100_results_idx],
                                                y_labels=client_labels[top_100_results_idx],
                                                matrix_values=top_100_results_matrix)})
