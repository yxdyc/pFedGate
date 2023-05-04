import numpy as np
import torch
import itertools
from numba import jit


class KnapsackSolver01(object):
    """
    A knapsack problem solver implementation for 0-1 Knapsack with large Weights,
    ref: https://www.geeksforgeeks.org/knapsack-with-large-weights/

    time complexity: O(value_sum_max * item_num_max) = O(item_num_max * item_num_max) in our setting
    auxiliary space: O(value_sum_max * item_num_max) = O(item_num_max * item_num_max) in our setting
    """

    def __init__(self, value_sum_max, item_num_max, weight_max):
        self.value_sum_max = value_sum_max
        self.item_num_max = item_num_max
        self.weight_max = weight_max

        # dp[V][i] represents the minimum weight subset of the subarray arr[i,...,N-1]
        # required to get a value of at least V.
        self.dp = np.zeros((value_sum_max + 1, item_num_max))
        self.value_solved = np.zeros((value_sum_max + 1, item_num_max))
        self.selected_item = np.zeros(item_num_max)

    def reset_state(self, value_sum_max=0, item_num_max=0, weight_max=0, iter_version=False):
        self.value_sum_max = self.value_sum_max if value_sum_max == 0 else value_sum_max
        self.item_num_max = self.item_num_max if item_num_max == 0 else item_num_max
        self.weight_max = self.weight_max if weight_max == 0 else weight_max
        self.selected_item = np.zeros(item_num_max)

        if not iter_version:
            self.value_solved = np.zeros((self.value_sum_max + 1, self.item_num_max))
            self.dp = np.zeros((self.value_sum_max + 1, self.item_num_max))
        else:
            self.dp = np.full((self.value_sum_max + 1, self.item_num_max), -1)
            self.selected_item = np.zeros((self.value_sum_max + 1, self.item_num_max))

    # Function to solve the recurrence relation
    def solve_dp(self, r, i, w, val, n):
        # Base cases
        if r <= 0:
            return 0
        if i == n:
            return self.weight_max
        if self.value_solved[r][i]:
            return self.dp[r][i]

        # Marking state as solved
        self.value_solved[r][i] = 1

        # Recurrence relation.  the maximum recursive depth is n
        # self.dp[r][i] = min(self.solve_dp(r, i + 1, w, val, n),
        #                     w[i] + self.solve_dp(r - val[i], i + 1, w, val, n))
        w_discard_item_i = self.solve_dp(r, i + 1, w, val, n)
        # r - val[i] indicates the value must be from item i, i.e., select the item
        w_hold_item_i = w[i] + self.solve_dp(r - val[i], i + 1, w, val, n)

        if w_discard_item_i < w_hold_item_i:
            self.dp[r][i] = w_discard_item_i
            self.selected_item[i] = 0
        else:
            self.dp[r][i] = w_hold_item_i
            self.selected_item[i] = 1

        return self.dp[r][i]

    def found_max_value(self, weight_list, value_list, capacity):
        value_sum_max = int(sum(value_list))
        weight_max = int(sum(weight_list))
        # weight_max = capacity  # the 86.7 version, while always select the last ones
        self.reset_state(value_sum_max=value_sum_max, item_num_max=len(weight_list), weight_max=weight_max)
        # Iterating through all possible values
        # to find the the largest value that can
        # be represented by the given weights and capacity constraints
        for i in range(value_sum_max, -1, -1):
            res = self.solve_dp(i, 0, weight_list, value_list, self.item_num_max)
            if res <= capacity:
                return i, np.nonzero(self.selected_item)

        return 0, np.nonzero(self.selected_item)

    def found_max_value_greedy(self, weight_list, value_list, capacity):
        if isinstance(value_list, np.ndarray):
            sorted_idx = (-value_list).argsort()[:len(value_list)]
        elif isinstance(value_list, torch.Tensor):
            sorted_value_per_weight, sorted_idx = value_list.sort(descending=True)
        else:
            raise NotImplementedError(
                f"found_max_value_greedy, only support value_list as ndarray or Tensor, while got {type(value_list)}")
        selected_idx = []
        droped_idx = []
        total_weight = 0
        total_value = 0
        for idx in sorted_idx:
            weight_after_select = total_weight + weight_list[idx]
            if weight_after_select <= capacity:
                total_weight = weight_after_select
                total_value += value_list[idx]
                selected_idx.append(idx)
            else:
                droped_idx.append(idx)

        return total_value, total_weight, selected_idx, droped_idx


class ItemUnitCost:
    def __init__(self, wt, val, ind):
        self.wt = wt
        self.val = val
        self.ind = ind
        self.cost = val / wt

    def __lt__(self, other):
        return self.cost < other.cost


class KnapsackSolverFractional(object):
    """
    A knapsack problem solver implementation for fractional Knapsack,
    in which each item can be selected with a sub-part.
    ref: https://home.cse.ust.hk/~dekai/271/notes/L14/L14.pdf

    # Greedy Approach
    time complexity: O(N*logN) as the solution is based on sort
    """

    def __init__(self, item_num_max, weight_max):
        self.item_num_max = item_num_max
        self.weight_max = weight_max

        # stored the results
        self.selected_item = np.zeros(item_num_max)

    def reset_state(self, item_num_max, weight_max):
        self.item_num_max = item_num_max
        self.weight_max = weight_max
        self.selected_item = np.zeros(item_num_max)

    def _found_max_value_numeric(self, weight_list, value_list, capacity):
        self.selected_item = np.zeros(self.item_num_max)
        iVal = []
        for i in range(len(weight_list)):
            iVal.append(ItemUnitCost(weight_list[i], value_list[i], i))
        iVal.sort(reverse=True)
        totalValue = 0
        for item in iVal:
            curWt = item.wt
            curVal = item.val
            # select the whole item
            if capacity - curWt >= 0:
                capacity -= curWt
                self.selected_item[item.ind] = curWt
                totalValue += curVal
            # select sub-set of the item
            else:
                fraction = capacity / curWt
                self.selected_item[item.ind] = capacity
                totalValue += curVal * fraction
                capacity = capacity - (curWt * fraction)
                break
        return totalValue, self.selected_item, self.selected_item / np.array(weight_list)

    def _found_max_value_tensor(self, weight_list, value_list, capacity):
        self.selected_item = torch.zeros_like(weight_list, device=weight_list.device)

        value_per_weight = value_list / weight_list
        sorted_value_per_weight, sorted_idx = value_per_weight.sort(descending=True)

        total_val = 0
        for i, unit_value in enumerate(sorted_value_per_weight):
            cur_item_idx = sorted_idx[i]
            cur_weight = weight_list[cur_item_idx]
            cur_val = value_list[cur_item_idx]
            # select the whole item
            if capacity - cur_weight >= 0:
                capacity -= cur_weight
                self.selected_item[cur_item_idx] = cur_weight
                total_val += cur_val
            # select sub-set of the item
            else:
                fraction = capacity / cur_weight
                self.selected_item[cur_item_idx] = capacity
                total_val += cur_val * fraction
                capacity = capacity - (cur_weight * fraction)
                break
        return total_val, self.selected_item, self.selected_item / weight_list

    def found_max_value(self, weight_list, value_list, capacity):
        """function to get maximum value """
        self.reset_state(item_num_max=len(weight_list), weight_max=capacity)

        tensor_type = isinstance(weight_list, torch.Tensor) and isinstance(value_list, torch.Tensor)
        normal_numeric_type = isinstance(weight_list[0], (int, float, complex)) and isinstance(value_list[0],
                                                                                               (int, float, complex))
        assert tensor_type or normal_numeric_type, \
            f"Unsupported weight_list: {type(weight_list)}, value_list: {type(value_list)}"

        if tensor_type:
            res = self._found_max_value_tensor(weight_list, value_list, capacity)
        else:
            res = self._found_max_value_numeric(weight_list, value_list, capacity)
        return res
