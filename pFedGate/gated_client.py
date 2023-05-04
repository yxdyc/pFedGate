import logging

import torch
import wandb

from client import Client


class pFedGateClient(Client):
    r"""
    Implements client for the proposed pFedGate method
    """

    def __init__(
            self,
            learners_ensemble,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            tune_locally=False,
            node_id=-1,
            bi_level_opt=1
    ):
        super(pFedGateClient, self).__init__(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
        )
        self.bi_level_opt = bi_level_opt
        self.val_loader = iter(self.val_iterator) if self.val_iterator is not None else None
        self.test_loader = iter(self.test_iterator)
        self.node_id = node_id
        self.learners_ensemble.learners[0].node_id = node_id

    def cache_model(self):
        learner = self.learners_ensemble.learners[0]
        learner.model.to("cpu")

    def restore_cache(self):
        learner = self.learners_ensemble.learners[0]
        learner.model.to(learner.device)

    def step(self, local_step_as_batch=False, *args, **kwargs):
        """
        perform on step for the client

        :param local_step_as_batch: if true, the client uses local_step batches to perform the update
        :return
            clients_updates: ()
        """
        assert len(self.learners_ensemble.learners) == 1, "pFedGate supports only 1 learner now"
        learner = self.learners_ensemble.learners[0]

        self.counter += 1

        total_loss_meta_model = 0
        total_metric = 0
        total_sparse_ratio_selected = 0
        total_top_gating_weights = torch.zeros(len(learner.gating_layer.block_size_lookup_table), device=learner.device)
        trained_sample_n = 0

        for i in range(self.local_steps):
            if local_step_as_batch:
                # bi-level optimization via an iterative manner
                batch = self.get_next_batch(dataset_type="train")
                trained_sample_n += batch[0].size(0)
                if self.bi_level_opt == 1:
                    loss_meta_model, metric, sparse_ratio_selected, top_gating_weights = learner.fit_batch(
                        batch=batch,
                        train_gating_layer=True,
                        train_base_model=False
                    )
                    loss_meta_model, metric, sparse_ratio_selected, top_gating_weights = learner.fit_batch(
                        batch=batch,
                        train_gating_layer=False,
                        train_base_model=True
                    )
                elif self.bi_level_opt == 0:
                    loss_meta_model, metric, sparse_ratio_selected, top_gating_weights = learner.fit_batch(
                        batch=batch,
                        train_gating_layer=True,
                        train_base_model=True
                    )
                else:
                    raise NotImplementedError("Unsupported gating optimization mode")
            else:
                if self.bi_level_opt == 1:
                    loss_meta_model, metric, sparse_ratio_selected, top_gating_weights, n_samples = \
                        learner.fit_epoch(
                            iterator=self.train_iterator,
                            # val_iter
                            train_gating_layer=True,
                            train_base_model=False
                        )
                    loss_meta_model, metric, sparse_ratio_selected, top_gating_weights, n_samples = \
                        learner.fit_epoch(
                            iterator=self.train_iterator,
                            train_gating_layer=False,
                            train_base_model=True
                        )
                elif self.bi_level_opt == 0:
                    loss_meta_model, metric, sparse_ratio_selected, top_gating_weights, n_samples = \
                        learner.fit_epoch(
                            iterator=self.train_iterator,
                            train_gating_layer=True,
                            train_base_model=True
                        )
                elif self.bi_level_opt == 2:  # + epoch-level adapt
                    # first sample_level_adapt to train the gating layer
                    loss_meta_model, metric, sparse_ratio_selected, top_gating_weights, n_samples = \
                        learner.fit_epoch(
                            iterator=self.train_iterator,
                            # val_iter
                            train_gating_layer=True,
                            train_base_model=False,
                            sample_level_adapt=True
                        )
                    # then client_level_adapt to train the base model
                    learner.client_level_adapt(iterator=self.train_iterator, train_gating_layer=False,
                                               train_base_model=True, )
                    loss_meta_model, metric, sparse_ratio_selected, top_gating_weights, n_samples = \
                        learner.fit_epoch(
                            iterator=self.train_iterator,
                            train_gating_layer=False,
                            train_base_model=True,
                            sample_level_adapt=False
                        )
                    pass
                else:
                    raise NotImplementedError("Unsupported gating optimization mode")
                trained_sample_n += n_samples
            total_loss_meta_model += loss_meta_model
            total_metric += metric
            total_sparse_ratio_selected += sparse_ratio_selected
            total_top_gating_weights += top_gating_weights

        total_loss_meta_model /= self.local_steps
        total_metric /= self.local_steps
        total_sparse_ratio_selected /= self.local_steps
        total_top_gating_weights /= self.local_steps

        # TODO: add flag arguments to use `free_gradients`
        self.learners_ensemble.free_gradients()

        return total_loss_meta_model, total_metric, total_sparse_ratio_selected, total_top_gating_weights, trained_sample_n

    def get_next_batch(self, dataset_type="train"):
        if dataset_type == "train":
            loader = self.train_loader
            itera = self.train_iterator
        elif dataset_type == "val":
            loader = self.val_loader
            itera = self.val_iterator
        elif dataset_type == "test":
            loader = self.test_loader
            itera = self.test_iterator
        else:
            raise ValueError(f"not supported dataset_type: {dataset_type}")

        try:
            batch = next(loader)
        except StopIteration:
            loader = iter(itera)
            batch = next(loader)

        return batch

    def write_logs(self, test_iter=None, val_iter=None, node_setting=None):
        if test_iter is None:
            test_iter = self.test_iterator
        if val_iter is None:
            val_iter = self.val_iterator
        if node_setting is None:
            node_setting = f"node{self.node_id}"
        if self.tune_locally:
            self.update_tuned_learners()
            eval_learners = self.tuned_learners_ensemble
        else:
            eval_learners = self.learners_ensemble

        assert len(eval_learners.learners) == 1, "pFedGate supports only 1 learner now"
        eval_learner = eval_learners.learners[0]

        # test
        sample_level_adapt = False if self.bi_level_opt == 2 else True
        if not sample_level_adapt and self.counter == 0:
            eval_learner.client_level_adapt(iterator=self.train_iterator, train_gating_layer=False,
                                            train_base_model=False, )
        test_loss_meta_model, test_metric, test_sparse_ratio_selected, test_top_gating_weights, test_n_samples = \
            eval_learner.fit_epoch(
                iterator=test_iter,
                train_gating_layer=False,
                train_base_model=False,
                sample_level_adapt=sample_level_adapt
            )
        # val
        if val_iter is not None:
            val_loss_meta_model, val_metric, val_sparse_ratio_selected, val_top_gating_weights, val_n_samples = \
                eval_learner.fit_epoch(
                    iterator=val_iter,
                    train_gating_layer=False,
                    train_base_model=False,
                    sample_level_adapt=sample_level_adapt
                )
        else:
            val_loss_meta_model = 0
            val_metric = 0
            val_sparse_ratio_selected = 0
            val_top_gating_weights = torch.zeros_like(test_top_gating_weights)
            val_n_samples = 0
        try:
            # commit=False makes the global step of wandb not incremented, such that we can log for different nodes
            wandb.log({f"{node_setting}/val/loss_meta_model": val_loss_meta_model,
                       f"{node_setting}/val/sparse_ratio_selected": val_sparse_ratio_selected,
                       f"{node_setting}/val/metric": val_metric,
                       f"{node_setting}/test/loss_meta_model": test_loss_meta_model,
                       f"{node_setting}/test/sparse_ratio_selected": test_sparse_ratio_selected,
                       f"{node_setting}/test/metric": test_metric,
                       f"{node_setting}/test/client_counter": self.counter}, commit=True)
        except ModuleNotFoundError:
            logging.warning("not found wandb, will not track wandb related results")

        return val_loss_meta_model, val_metric, val_sparse_ratio_selected, val_top_gating_weights, \
               test_loss_meta_model, test_metric, test_sparse_ratio_selected, test_top_gating_weights
