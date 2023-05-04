import shutil
from typing import Callable, Optional

from wandb.sdk.lib import filenames
from wandb.sdk.lib.filenames import WANDB_DIRS

from models.adapted_op import AdaptedLeafCNN1, AdaptedLeNet, AdaptedLeafCNN3
from pFedGate.gate_aggregator import pFedGateAggregator
from pFedGate.gated_learner import GatedLearner
from pFedGate.gated_client import pFedGateClient
from models.gating_layers import GatingLayer
from models.nn_nets import *
from datasets import *
from learners.learner import *
from learners.learners_ensemble import *
from client import *
from aggregator import *

from .optim import *
from .metrics import *
from .constants import *
from .decentralized import *

from torch.utils.data import DataLoader
import tempfile
import tarfile

from tqdm import tqdm

import math

from .sparse_factor_schedule import SparsityLinearScheduler, SparsityMultiStepScheduler, \
    SparsityReduceLROnPlateauScheduler


def get_data_dir(experiment_name):
    """
    returns a string representing the path where to find the datafile corresponding to the experiment

    :param experiment_name: name of the experiment
    :return: str

    """
    if "." in experiment_name:
        name, sub_name = experiment_name.split(".")
        data_dir = os.path.join("data", name, sub_name, "all_data")
    else:
        data_dir = os.path.join("data", experiment_name, "all_data")

    return data_dir


def get_sparse_factor_scheduler(scheduler_name, n_rounds, s_target):
    if scheduler_name == "constant":
        return None
    elif scheduler_name == "linear":
        # similar ref <<To prune, or not to prune: exploring the efficacy of pruning for model compression>>
        prune_begin_round = 0.2 * n_rounds
        total_decay_rounds = 0.5 * n_rounds
        return SparsityLinearScheduler(prune_begin_round=prune_begin_round, total_rounds=total_decay_rounds,
                                       s_target=s_target, s_begin=1)
    elif scheduler_name == "multi_step":
        decay_times = 4
        decay_each_time = (1 - s_target) / decay_times
        total_decay_rounds = n_rounds * 0.8
        decay_round_delta = total_decay_rounds / decay_times
        s_list = [1] + [1 - decay_each_time * (i + 1) for i in range(decay_times)] + [s_target]
        round_list = [0] + [int(decay_round_delta * (i + 1)) for i in range(decay_times)] + [int(n_rounds)]
        return SparsityMultiStepScheduler(change_round_list=round_list, s_list=s_list)
    elif scheduler_name == "reduce_on_plateau":
        decay_times = 4
        decay_factor_bound = math.pow(s_target, 1 / decay_times)
        return SparsityReduceLROnPlateauScheduler(decay_factor=decay_factor_bound, init_s=1, min_sparse_factor=s_target,
                                                  mode="max", patience=10)
    else:
        raise NotImplementedError(f"Unsupported sparse_factor_scheduler: {scheduler_name}")


def get_learner(
        name,
        device,
        optimizer_name,
        scheduler_name,
        initial_lr,
        mu,
        n_rounds,
        seed,
        input_dim=None,
        output_dim=None,
        gated_learner=False,
        alpha=0,
        beta=0,
        sparse_factor=1,
        block_wise_prune=False,
        fine_grained_block_split=1,
        sparse_factor_scheduler="constant",
        track_running_stats=1,
        shared_model=None,
        args_=None
):
    """
    constructs the learner corresponding to an experiment for a given seed

    :param name: name of the experiment to be used; possible are
                 {`synthetic`, `cifar10`, `emnist`, `shakespeare`}
    :param device: used device; possible `cpu` and `cuda`
    :param optimizer_name: passed as argument to utils.optim.get_optimizer
    :param scheduler_name: passed as argument to utils.optim.get_lr_scheduler
    :param initial_lr: initial value of the learning rate
    :param mu: proximal term weight, only used when `optimizer_name=="prox_sgd"`
    :param input_dim: input dimension, only used for synthetic dataset
    :param output_dim: output_dimension; only used for synthetic dataset
    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`, default is None;
    :param seed:
    :return: Learner

    """
    torch.manual_seed(seed)

    criterion, is_binary_classification, metric, model = get_model(device, input_dim, name, output_dim, shared_model,
                                                                   model_type=args_.model_type)

    if isinstance(initial_lr, list):
        lr_model, lr_gating = initial_lr
    else:
        lr_model = lr_gating = initial_lr
    optimizer = \
        get_optimizer(
            optimizer_name=optimizer_name,
            model=model,
            lr_initial=lr_model,
            mu=mu
        )
    lr_scheduler = \
        get_lr_scheduler(
            optimizer=optimizer,
            scheduler_name=scheduler_name,
            n_rounds=n_rounds,
            gated_learner=gated_learner,
        )
    sparse_factor_scheduler = get_sparse_factor_scheduler(
        scheduler_name=sparse_factor_scheduler, n_rounds=n_rounds, s_target=sparse_factor)

    if not gated_learner:
        if name == "shakespeare":
            return LanguageModelingLearner(
                model=model,
                criterion=criterion,
                metric=metric,
                device=device,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                is_binary_classification=is_binary_classification
            )
        else:
            return Learner(
                model=model,
                criterion=criterion,
                metric=metric,
                device=device,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                is_binary_classification=is_binary_classification
            )
    else:
        # set up learned gating layer and its optimizer, lr_scheduler
        gating_layer = GatingLayer(model_to_mask=model, device=device, dataset_name=name,
                                   fine_grained_block_split=fine_grained_block_split,
                                   seperate_trans=args_.seperate_trans)
        opt_for_gating = \
            get_optimizer(
                optimizer_name=optimizer_name,
                model=gating_layer,
                lr_initial=lr_gating,
                mu=mu
            )
        lr_scheduler_for_gating = \
            get_lr_scheduler(
                optimizer=opt_for_gating,
                scheduler_name=scheduler_name,
                n_rounds=n_rounds,
                gated_learner=gated_learner
            )

        return GatedLearner(
            base_model=model,
            criterion=criterion,
            metric=metric,
            device=device,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_binary_classification=is_binary_classification,
            gating_layer=gating_layer,
            opt_for_gating=opt_for_gating,
            lr_scheduler_for_gating=lr_scheduler_for_gating,
            alpha=alpha,
            beta=beta,
            sparse_factor=sparse_factor,
            block_wise_prune=block_wise_prune,
            sparse_factor_scheduler=sparse_factor_scheduler,
            bn_running_stats=track_running_stats,
            args_=args_
        )


def get_model(device, input_dim, name, output_dim, shared_model=None, model_type=None):
    model = shared_model
    if name == "synthetic":
        if output_dim == 2:
            criterion = nn.BCEWithLogitsLoss(reduction="none").to(device)
            metric = binary_accuracy
            if not shared_model:
                model = LinearLayer(input_dim, 1).to(device)
            is_binary_classification = True
        else:
            criterion = nn.CrossEntropyLoss(reduction="none").to(device)
            metric = accuracy
            if not shared_model:
                model = LinearLayer(input_dim, output_dim).to(device)
            is_binary_classification = False
    elif name == "cifar10":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        if not shared_model:
            # model = get_mobilenet(n_classes=10).to(device)
            # model = get_resnet18(n_classes=10).to(device)
            model = AdaptedLeNet(num_classes=10).to(device)
        is_binary_classification = False
    elif name == "cifar100":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        if not shared_model:
            if model_type == "lenet5":
                model = AdaptedLeNet(num_classes=100, n_kernels=32, in_channels=3, fc_factor=1).to(device)
            elif model_type == "lenet5_wide":
                model = AdaptedLeNet(num_classes=100, n_kernels=32 * 4, in_channels=3, fc_factor=4).to(device)
            elif model_type == "lenet5_fc_wide":
                model = AdaptedLeNet(num_classes=100, n_kernels=32, in_channels=3, fc_factor=16, fc_factor2=6).to(
                    device)
            elif model_type == "cnn_leaf":
                model = AdaptedLeafCNN3(num_classes=100).to(device)
            elif model_type == "cnn_leaf_wide":
                model = AdaptedLeafCNN3(num_classes=100, n_kernels=32 * 2, in_channels=3, fc_factor=2).to(device)
            else:
                model = AdaptedLeNet(num_classes=100, n_kernels=32, in_channels=3, fc_factor=1).to(device)
        is_binary_classification = False
    elif name == "emnist" or name == "femnist":
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
        metric = accuracy
        if not shared_model:
            model = AdaptedLeafCNN1(num_classes=62).to(device)
        is_binary_classification = False
    else:
        raise NotImplementedError
    return criterion, is_binary_classification, metric, model


def get_learners_ensemble(
        n_learners,
        name,
        device,
        optimizer_name,
        scheduler_name,
        initial_lr,
        mu,
        n_rounds,
        seed,
        input_dim=None,
        output_dim=None,
        gated_learner=False,
        alpha=0,
        beta=0,
        sparse_factor=1,
        block_wise_prune=False,
        fine_grained_block_split=1,
        sparse_factor_scheduler="constant",
        track_running_stats=1,
        shared_model=None,
        args_=None
):
    """
    constructs the learner corresponding to an experiment for a given seed

    :param n_learners: number of learners in the ensemble
    :param name: name of the experiment to be used; possible are
                 {`synthetic`, `cifar10`, `emnist`, `shakespeare`}
    :param device: used device; possible `cpu` and `cuda`
    :param optimizer_name: passed as argument to utils.optim.get_optimizer
    :param scheduler_name: passed as argument to utils.optim.get_lr_scheduler
    :param initial_lr: initial value of the learning rate
    :param mu: proximal term weight, only used when `optimizer_name=="prox_sgd"`
    :param input_dim: input dimension, only used for synthetic dataset
    :param output_dim: output_dimension; only used for synthetic dataset
    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`, default is None;
    :param seed:
    :return: LearnersEnsemble

    """
    learners = [
        get_learner(
            name=name,
            device=device,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            initial_lr=initial_lr,
            input_dim=input_dim,
            output_dim=output_dim,
            n_rounds=n_rounds,
            seed=seed + learner_id,
            mu=mu,
            gated_learner=gated_learner,
            alpha=alpha,
            beta=beta,
            sparse_factor=sparse_factor,
            block_wise_prune=block_wise_prune,
            fine_grained_block_split=fine_grained_block_split,
            sparse_factor_scheduler=sparse_factor_scheduler,
            track_running_stats=track_running_stats,
            shared_model=shared_model,
            args_=args_
        ) for learner_id in range(n_learners)
    ]

    learners_weights = torch.ones(n_learners) / n_learners
    if name == "shakespeare":
        return LanguageModelingLearnersEnsemble(learners=learners, learners_weights=learners_weights)
    else:
        return LearnersEnsemble(learners=learners, learners_weights=learners_weights)


def get_loaders(type_, root_path, batch_size, is_validation):
    """
    constructs lists of `torch.utils.DataLoader` object from the given files in `root_path`;
     corresponding to `train_iterator`, `val_iterator` and `test_iterator`;
     `val_iterator` iterates on the same dataset as `train_iterator`, the difference is only in drop_last

    :param type_: type of the dataset;
    :param root_path: path to the data folder
    :param batch_size:
    :param is_validation: (bool) if `True` validation part is used as test
    :return:
        train_iterator, val_iterator, test_iterator
        (List[torch.utils.DataLoader], List[torch.utils.DataLoader], List[torch.utils.DataLoader])

    """
    if type_ == "cifar10":
        inputs, targets = get_cifar10()
    elif type_ == "cifar100":
        inputs, targets = get_cifar100()
    elif type_ == "emnist":
        inputs, targets = get_emnist()
    else:
        inputs, targets = None, None

    train_iterators, val_iterators, test_iterators = {}, {}, {}

    # IO-intensive init since data loader indices are stored in many small (pickle) files,
    # set PARALLEL_INIT as true to speed up the initialization via multi-processing
    PARALLEL_INIT = True

    if not PARALLEL_INIT:
        for _, task_dir in enumerate(tqdm(os.listdir(root_path))):
            test_iterator, train_iterator, val_iterator, task_id = init_iterators_for_one_client(batch_size, inputs,
                                                                                                 is_validation,
                                                                                                 root_path,
                                                                                                 targets, task_dir,
                                                                                                 type_)

            train_iterators[task_id] = train_iterator
            val_iterators[task_id] = val_iterator
            test_iterators[task_id] = test_iterator
    else:
        from concurrent.futures import ThreadPoolExecutor
        torch.multiprocessing.set_sharing_strategy('file_system')
        import resource
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        assert 1048576 <= rlimit[1]
        resource.setrlimit(resource.RLIMIT_NOFILE, (1048576, rlimit[1]))

        def loader_init_callback_multi_process(ret):
            test_iterator, train_iterator, val_iterator, task_id = ret
            train_iterators[task_id] = train_iterator
            val_iterators[task_id] = val_iterator
            test_iterators[task_id] = test_iterator

        def loader_init_callback_multi_thread(ret):
            test_iterator, train_iterator, val_iterator, task_id = ret.result()
            train_iterators[task_id] = train_iterator
            val_iterators[task_id] = val_iterator
            test_iterators[task_id] = test_iterator

        with ThreadPoolExecutor() as pool:
            # with ProcessPoolExecutor() as pool:
            for _, task_dir in enumerate(tqdm(os.listdir(root_path))):
                future = pool.submit(init_iterators_for_one_client, batch_size, inputs,
                                     is_validation, root_path,
                                     targets, task_dir, type_)
                future.add_done_callback(loader_init_callback_multi_thread)

    return train_iterators, val_iterators, test_iterators


def init_iterators_for_one_client(batch_size, inputs, is_validation, root_path, targets, task_dir, type_):
    task_data_path = os.path.join(root_path, task_dir)
    task_id = task_dir.split("_")[-1]
    logging.info(f"BEGIN: init iterators for task {task_id}")
    train_iterator = \
        get_loader(
            type_=type_,
            path=os.path.join(task_data_path, f"train{EXTENSIONS[type_]}"),
            batch_size=batch_size,
            inputs=inputs,
            targets=targets,
            train=True
        )
    if os.path.exists(os.path.join(task_data_path, f"val{EXTENSIONS[type_]}")):
        val_iterator = \
            get_loader(
                type_=type_,
                path=os.path.join(task_data_path, f"val{EXTENSIONS[type_]}"),
                batch_size=batch_size,
                inputs=inputs,
                targets=targets,
                train=False
            )
    else:
        val_iterator = None
    if is_validation:
        test_set = "val"
    else:
        test_set = "test"
    test_iterator = \
        get_loader(
            type_=type_,
            path=os.path.join(task_data_path, f"{test_set}{EXTENSIONS[type_]}"),
            batch_size=batch_size,
            inputs=inputs,
            targets=targets,
            train=False
        )
    logging.info(f"END: init iterators for task {task_id}")
    return test_iterator, train_iterator, val_iterator, task_id


def get_loader(type_, path, batch_size, train, inputs=None, targets=None):
    """
    constructs a torch.utils.DataLoader object from the given path

    :param type_: type of the dataset; possible are `tabular`, `images` and `text`
    :param path: path to the data file
    :param batch_size:
    :param train: flag indicating if train loader or test loader
    :param inputs: tensor storing the input data; only used with `cifar10`, `cifar100` and `emnist`; default is None
    :param targets: tensor storing the labels; only used with `cifar10`, `cifar100` and `emnist`; default is None
    :return: torch.utils.DataLoader

    """
    if type_ == "tabular":
        dataset = TabularDataset(path)
    elif type_ == "cifar10":
        dataset = SubCIFAR10(path, cifar10_data=inputs, cifar10_targets=targets)
    elif type_ == "cifar100":
        dataset = SubCIFAR100(path, cifar100_data=inputs, cifar100_targets=targets)
    elif type_ == "emnist":
        dataset = SubEMNIST(path, emnist_data=inputs, emnist_targets=targets)
    elif type_ == "femnist":
        dataset = SubFEMNIST(path)
    elif type_ == "shakespeare":
        dataset = CharacterDataset(path, chunk_len=SHAKESPEARE_CONFIG["chunk_len"])
    else:
        raise NotImplementedError(f"{type_} not recognized type; possible are {list(LOADER_TYPE.keys())}")

    if len(dataset) == 0:
        return

    # drop last batch, because of BatchNorm layer used in mobilenet_v2
    drop_last = ((type_ == "cifar100") or (type_ == "cifar10")) and (len(dataset) > batch_size) and train

    g = torch.Generator()
    g.manual_seed(1234)

    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=drop_last, generator=g)


def log_codes(
        wandb_run,
        outdir,
        log_to_wandb=False,  # Error 411
        root: str = ".",
        name: str = None,
        include_fn: Callable[[str], bool] = lambda path: path.endswith(".py"),
        exclude_fn: Callable[[str], bool] = filenames.exclude_wandb_fn,
) -> Optional[wandb.Artifact]:
    """Saves the current state of your code to a W&B artifact.

    By default it walks the current directory and logs all files that end with `.py`.

    Arguments:
        root (str, optional): The relative (to `os.getcwd()`) or absolute path to
            recursively find code from.
        name (str, optional): The name of our code artifact. By default we'll name
            the artifact `source-$RUN_ID`. There may be scenarios where you want
            many runs to share the same artifact. Specifying name allows you to achieve that.
        include_fn (callable, optional): A callable that accepts a file path and
            returns True when it should be included and False otherwise. This
            defaults to: `lambda path: path.endswith(".py")`
        exclude_fn (callable, optional): A callable that accepts a file path and
            returns `True` when it should be excluded and `False` otherwise. This
            defaults to: `lambda path: False`

    Examples:
        Basic usage
        ```python
        run.log_code()
        ```

        Advanced usage
        ```python
        run.log_code("../", include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb"))
        ```

    Returns:
        An `Artifact` object if code was logged
    """
    name = name or "{}-{}".format("source", wandb_run.id)
    if log_to_wandb:
        art = wandb.Artifact(name, "code")
    files_added = False
    temp = tempfile.NamedTemporaryFile('wb', suffix='.tar.gz', delete=False)
    tar = tarfile.open(fileobj=temp, mode='w:gz')
    if root is not None:
        root = os.path.abspath(root)
        for file_path in filtered_dir(root, include_fn, exclude_fn):
            files_added = True
            save_name = os.path.relpath(file_path, root)
            if log_to_wandb:
                art.add_file(file_path, name=save_name)
            tar.add(file_path, arcname=save_name)
    tar.close()
    shutil.copyfile(temp.name, os.path.join(outdir, name + '-code.tar.gz'))

    # Add any manually staged files such is ipynb notebooks
    for dirpath, _, files in os.walk(wandb_run._settings._tmp_code_dir):
        for fname in files:
            file_path = os.path.join(dirpath, fname)
            save_name = os.path.relpath(file_path, wandb_run._settings._tmp_code_dir)
            files_added = True
            if log_to_wandb:
                art.add_file(file_path, name=save_name)
    if not files_added:
        return None
    if log_to_wandb:
        wandb_run.log_artifact(art)
    return


def filtered_dir(
        root: str, include_fn: Callable[[str], bool], exclude_fn: Callable[[str], bool]
):
    """Simple generator to walk a directory"""
    for dirpath, dirs, files in os.walk(root, topdown=True):
        dirs[:] = [d for d in dirs if not exclude_fn(d)]
        for fname in files:
            file_path = os.path.join(dirpath, fname)
            if include_fn(file_path) and not exclude_fn(file_path):
                yield file_path


def exclude_wandb_fn(path: str) -> bool:
    return any(os.sep + wandb_dir + os.sep in path for wandb_dir in WANDB_DIRS)


def get_client(
        client_type,
        learners_ensemble,
        q,
        train_iterator,
        val_iterator,
        test_iterator,
        logger,
        local_steps,
        tune_locally,
        node_id=0,
        bi_level_opt=1
):
    """

    :param client_type:
    :param learners_ensemble:
    :param q: fairness hyper-parameter, ony used for FFL client
    :param train_iterator:
    :param val_iterator:
    :param test_iterator:
    :param logger:
    :param local_steps:
    :param tune_locally

    :return:

    """
    if client_type == "mixture":
        return MixtureClient(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            node_id=node_id,
            tune_locally=tune_locally
        )
    elif client_type == "AFL":
        return AgnosticFLClient(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally
        )
    elif client_type == "FFL":
        return FFLClient(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            q=q
        )
    elif client_type == "pFedGate":
        return pFedGateClient(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            node_id=node_id,
            bi_level_opt=bi_level_opt
        )
    else:
        return Client(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally
        )


def get_aggregator(
        aggregator_type,
        clients,
        global_learners_ensemble,
        lr,
        lr_lambda,
        mu,
        communication_probability,
        q,
        sampling_rate,
        log_freq,
        global_train_logger,
        global_test_logger,
        test_clients,
        verbose,
        seed=None,
        aggregate_sampled_clients=0,
        online_aggregate=0,
        outdir=""
):
    """
    `personalized` corresponds to pFedMe

    :param aggregator_type:
    :param clients:
    :param global_learners_ensemble:
    :param lr: oly used with FLL aggregator
    :param lr_lambda: only used with Agnostic aggregator
    :param mu: penalization term, only used with L2SGD
    :param communication_probability: communication probability, only used with L2SGD
    :param q: fairness hyper-parameter, ony used for FFL client
    :param sampling_rate:
    :param log_freq:
    :param global_train_logger:
    :param global_test_logger:
    :param test_clients
    :param verbose: level of verbosity
    :param seed: default is None
    :return:

    """
    seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    if aggregator_type == "no_communication":
        return NoCommunicationAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "centralized":
        return CentralizedAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "personalized":
        return PersonalizedAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "clustered":
        return ClusteredAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            test_clients=test_clients,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "L2SGD":
        return LoopLessLocalSGDAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            communication_probability=communication_probability,
            penalty_parameter=mu,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "AFL":
        return AgnosticAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            test_clients=test_clients,
            lr_lambda=lr_lambda,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "FFL":
        return FFLAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            test_clients=test_clients,
            lr=lr,
            q=q,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "decentralized":
        n_clients = len(clients)
        mixing_matrix = get_mixing_matrix(n=n_clients, p=0.5, seed=seed)

        return DecentralizedAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            mixing_matrix=mixing_matrix,
            log_freq=log_freq,
            test_clients=test_clients,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed
        )
    elif aggregator_type == "pFedGate":
        return pFedGateAggregator(
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            log_freq=log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            sampling_rate=sampling_rate,
            verbose=verbose,
            seed=seed,
            aggregate_sampled_clients=aggregate_sampled_clients,
            online_aggregate=online_aggregate,
            outdir=outdir
        )
    else:
        raise NotImplementedError(
            "{aggregator_type} is not a possible aggregator type."
            " Available are: `no_communication`, `centralized`,"
            " `personalized`, `clustered`, `fednova`, `AFL`,"
            " `FFL` and `decentralized`."
        )


class RunningStats(object):

    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def reset(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        if isinstance(self.new_s, torch.Tensor):
            return torch.sqrt(self.variance())
        else:
            return math.sqrt(self.variance())
