from copy import copy, deepcopy

import torch
from torch import nn as nn

from models import switchable_norm
from models.adapted_op import AdaptedLinear, deepgetattr
from utils.constants import IN_PLANES_TYPE, SHAKESPEARE_CONFIG
from models.adapted_op import map_module_name


class Reshape(nn.Module):
    def __init__(self, reshape_type="flat_dim0"):
        super(Reshape, self).__init__()
        self.reshape_type = reshape_type

    def forward(self, x):
        if self.reshape_type == "flat_dim0":
            B = x.size()[0]
            return x.view(B, -1)
        else:
            raise NotImplementedError("Un-supported reshape_type: {}".format(self.reshape_type))


class GatingLayer(nn.Module):
    def __init__(self, model_to_mask, device, dataset_name, fine_grained_block_split=1, seperate_trans=0):
        super().__init__()
        assert dataset_name in ["shakespeare", "emnist", "femnist", "cifar10", "cifar100"], \
            f"Un-supported dataset name: {dataset_name}"

        # ----------------------------  Split Blocks into linear and non-linear parts  ----------------------------
        # model-block size, used for structured masking
        ori_block_names = [p[0] for p in model_to_mask.named_parameters()]
        ori_block_size_lookup_table = torch.tensor([p.numel() for p in model_to_mask.parameters()], device=device)
        # linear_layer_idx is used to select top masks for linear_layer and non-linear layer separately,
        # since linear layers usually have much larger parameters than other types
        linear_layer_block_idx = []
        for i, p in enumerate(model_to_mask.named_parameters()):
            para_name_second_last_level = ".".join(p[0].split(".")[:-1])
            para_name_second_last_level = map_module_name(para_name_second_last_level)
            para = deepgetattr(model_to_mask, para_name_second_last_level)
            if isinstance(para, torch.nn.Linear) or isinstance(para, AdaptedLinear):
                linear_layer_block_idx.append(i)

        self.fine_grained_block_split = fine_grained_block_split
        # a dict stores "para_name" -> (para_sub_block_begin_idx, para_sub_block_end_idx, size_each_sub)
        self.para_name_to_block_split_info = {}
        assert self.fine_grained_block_split >= 1, f"fine_grained_block_split must >= 1, while got {fine_grained_block_split}"
        if self.fine_grained_block_split == 1:
            self.block_names = ori_block_names
            self.block_size_lookup_table = ori_block_size_lookup_table
            self.linear_layer_block_idx = linear_layer_block_idx
        else:
            # each block is further split into fine_grained_block_split parts
            self.block_names = []
            self.block_size_lookup_table = []
            self.linear_layer_block_idx = []
            cur_block_idx = 0
            for i, name in enumerate(ori_block_names):
                # e.g., block_size = 25, fine_grained_block_split = 10
                split_num = self.fine_grained_block_split \
                    if ori_block_size_lookup_table[i] >= self.fine_grained_block_split \
                    else ori_block_size_lookup_table[i]
                # e.g., size_each_sub = ceil(25/10) = 3
                size_each_sub = torch.ceil(ori_block_size_lookup_table[i] / split_num).int()
                # e.g., split_num = ceil(25/3) = 9
                split_num = torch.ceil(ori_block_size_lookup_table[i] / size_each_sub).int()
                self.para_name_to_block_split_info[name] = (cur_block_idx, cur_block_idx + split_num, size_each_sub)
                for j in range(split_num - 1):
                    self.block_names.append(f"{name}.sub{j}")
                    self.block_size_lookup_table.append(size_each_sub)
                    if i in linear_layer_block_idx:
                        self.linear_layer_block_idx.append(cur_block_idx)
                    cur_block_idx += 1
                # for the last sub-block, in the case: (ori_block_size_lookup_table[i] % fine_grained_block_split !=0)
                self.block_names.append(f"{name}.sub{split_num - 1}")
                # e.g., size_last_sub = 25 - (9-1)*3 = 1
                size_last_sub = ori_block_size_lookup_table[i] - (split_num - 1) * size_each_sub
                self.block_size_lookup_table.append(size_last_sub)
                if i in linear_layer_block_idx:
                    self.linear_layer_block_idx.append(cur_block_idx)
                cur_block_idx += 1
        self.non_linear_layer_block_idx = [i for i in range(len(self.block_size_lookup_table))
                                           if i not in self.linear_layer_block_idx]

        # filter out the idx of first sub-blocks for each components,
        # to avoid cutting info flow (some internal sub-blocks are all zeros)
        self.linear_layer_block_idx_filter_first = []
        for idx in self.linear_layer_block_idx:
            if "sub0" not in self.block_names[idx]:
                self.linear_layer_block_idx_filter_first.append(idx)
        self.non_linear_layer_block_idx_filter_first = []
        for idx in self.non_linear_layer_block_idx:
            if "sub0" not in self.block_names[idx]:
                self.non_linear_layer_block_idx_filter_first.append(idx)

        if self.fine_grained_block_split != 1:
            self.block_size_lookup_table = torch.stack(self.block_size_lookup_table)

        block_size_lookup_table_linear = self.block_size_lookup_table[self.linear_layer_block_idx].double()
        block_size_lookup_table_non_linear = self.block_size_lookup_table[self.non_linear_layer_block_idx].double()
        block_size_lookup_table_linear /= block_size_lookup_table_linear.sum()
        block_size_lookup_table_non_linear /= block_size_lookup_table_non_linear.sum()
        self.block_size_lookup_table_normalized = deepcopy(self.block_size_lookup_table).double()
        self.block_size_lookup_table_normalized[self.linear_layer_block_idx] = block_size_lookup_table_linear
        self.block_size_lookup_table_normalized[self.non_linear_layer_block_idx] = block_size_lookup_table_non_linear

        # ---------------------------- Build gating layer ----------------------------

        if dataset_name is "shakespeare":
            norm_input_layer = switchable_norm.SwitchNorm1d(IN_PLANES_TYPE[dataset_name])
            input_feat_size = SHAKESPEARE_CONFIG["embed_size"]
        else:
            norm_input_layer = switchable_norm.SwitchNorm2d(IN_PLANES_TYPE[dataset_name])
            input_feat_size = 1 * 28 * 28 if dataset_name in ["emnist", "femnist"] else 3 * 32 * 32
        reshape_layer = Reshape(reshape_type="flat_dim0")
        output_layer = nn.Linear(in_features=input_feat_size, out_features=len(self.block_names))
        norm_outputs = torch.nn.BatchNorm1d(num_features=len(self.block_names))
        self.norm_input_layer = norm_input_layer
        self.gating = nn.Sequential(
            reshape_layer,
            output_layer,
            norm_outputs
        )

        self.seperate_trans = seperate_trans
        if self.seperate_trans == 1:
            self.w_transform = deepcopy(self.gating)
            self.w_transform.to(device)
        self.norm_input_layer.to(device)
        self.gating.to(device)

        self.norm_input_each_forward = True

    def forward(self, x):
        assert len(x.size()) in [3, 4], f"Un-expected input shape for gating layer, got {len(x.size())}"
        if self.norm_input_each_forward:
            x = self.norm_input_layer(x)
        gating_score = self._forward(x, self.gating)
        if self.seperate_trans == 1:
            trans_weight = self._forward(x, self.w_transform)
        else:
            trans_weight = gating_score
        return gating_score, trans_weight

    def _forward(self, x, layer):
        # predict suitable sub-blocks of base-model according to given example
        res = layer(x)
        if len(res.size()) == 3:  # [B, seq_len, block_len] for texts
            return torch.mean(res, dim=1)
        elif len(res.size()) == 2:  # [B, block_len] for images
            return res
        else:
            raise RuntimeError(f"Un-expected mask weights shape for gating layer, got {len(res.size())}")
