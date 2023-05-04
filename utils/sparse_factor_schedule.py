from torch._six import inf


class SparsityLinearScheduler(object):
    def __init__(self, prune_begin_round, total_rounds, s_target, s_begin):
        self.prune_begin_round = prune_begin_round
        self.total_decay_rounds = total_rounds
        self.s_target = s_target
        self.s_begin = s_begin
        # linear decay
        self.decay_each_round = (s_begin - s_target) / self.total_decay_rounds

        self.cur_s = s_begin
        self.cur_round = 0

    def step(self):
        if self.cur_round < self.prune_begin_round:
            self.cur_s = self.s_begin
        elif self.cur_round > (self.prune_begin_round + self.total_decay_rounds):
            self.cur_s = self.s_target
        else:
            self.cur_s -= self.decay_each_round
        self.cur_round += 1

        return self.cur_s


class SparsityMultiStepScheduler(object):
    def __init__(self, change_round_list, s_list):
        assert len(change_round_list) == len(s_list), \
            "You should specify the same length for change_round_list and s_list, e.g., [0, 100, 200], [1, 0.75, 0.5]." \
            f"Your input are {change_round_list} and {s_list}"
        assert change_round_list == (sorted(change_round_list)) and change_round_list[-1] >= change_round_list[0], \
            "change_round_list should be sorted in asc"
        self.change_round_list = change_round_list
        self.s_list = s_list

        self.cur_s = s_list[0]
        self.cur_round = 0
        self.s_begin = s_list[0]

    def step(self):
        self.cur_round += 1

        idx_in_list = 0
        while idx_in_list < (len(self.change_round_list) - 1):
            if self.change_round_list[idx_in_list] <= self.cur_round < self.change_round_list[idx_in_list + 1]:
                break
            else:
                idx_in_list += 1

        self.cur_s = self.s_list[idx_in_list]

        return self.cur_s


class SparsityReduceLROnPlateauScheduler(object):
    # ref: torch.optim.lr_scheduler.ReduceLROnPlateau
    def __init__(self, init_s, mode='min', decay_factor=0.9, patience=10,
                 threshold=1e-4, threshold_mode='rel', min_sparse_factor=0.1):
        if decay_factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = decay_factor

        self.patience = patience
        self.min_sparse_factor = min_sparse_factor
        self.cur_s = init_s
        self.s_begin = init_s
        self.cur_epoch = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_steps = None
        self.mode_worse = None  # the worse value for the chosen mode
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        current = float(metrics)
        if epoch is not None:
            self.cur_epoch = epoch
        else:
            self.cur_epoch += 1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_steps = 0
        else:
            self.num_bad_steps += 1

        if self.num_bad_steps > self.patience:
            self.cur_s *= self.factor
            if self.cur_s < self.min_sparse_factor:
                self.cur_s = self.min_sparse_factor
            self.num_bad_steps = 0

        return self.cur_s

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
