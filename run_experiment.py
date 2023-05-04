"""Run Experiment

This script allows to run one federated learning experiment; the experiment name, the method and the
number of clients/tasks should be precised along side with the hyper-parameters of the experiment.

The results of the experiment (i.e., training logs) are written to ./logs/ folder.

This file can also be imported as a module and contains the following function:

    * run_experiment - runs one experiments given its arguments
"""
import logging
import sys

from utils.utils import *
from utils.constants import *
from utils.args import *
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter


def init_clients(args_, root_path, logs_root, shared_model=None):
    """
    initialize clients from data folders
    :param args_:
    :param root_path: path to directory containing data folders
    :param logs_root: path to logs root
    :return: List[Client]
    """
    print("===> Building data iterators..")
    train_iterators, val_iterators, test_iterators = \
        get_loaders(
            type_=LOADER_TYPE[args_.dataset_name],
            root_path=root_path,
            batch_size=args_.bz,
            is_validation=args_.validation
        )

    print("===> Initializing clients..")
    clients_ = []
    total_train_num = 0
    total_val_num = 0
    total_test_num = 0
    for _, task_id in enumerate(tqdm(train_iterators.keys(), total=len(train_iterators))):
        train_iterator = train_iterators[task_id]
        val_iterator = val_iterators[task_id]
        test_iterator = test_iterators[task_id]
        if train_iterator is None or test_iterator is None:
            continue
        if val_iterator and val_iterator.dataset.indices == test_iterator.dataset.indices:
            val_iterator = None

        total_train_num += train_iterator.dataset.data.shape[0]
        total_val_num += val_iterator.dataset.data.shape[0] if val_iterator is not None else 0
        total_test_num += test_iterator.dataset.data.shape[0]

        learners_ensemble = \
            get_learners_ensemble(
                n_learners=args_.n_learners,
                name=args_.dataset_name,
                device=args_.device,
                optimizer_name=args_.optimizer,
                scheduler_name=args_.lr_scheduler,
                initial_lr=[args_.lr_model, args_.lr_gating],
                input_dim=args_.input_dimension,
                output_dim=args_.output_dimension,
                n_rounds=args_.n_rounds,
                seed=args_.seed,
                mu=args_.mu,
                gated_learner=(args_.method == 'pFedGate'),
                alpha=args_.alpha,
                beta=args_.beta,
                sparse_factor=args_.sparse_factor,
                block_wise_prune=(args_.block_wise_prune == 1),
                fine_grained_block_split=args_.fine_grained_block_split,
                sparse_factor_scheduler=args_.sparse_factor_scheduler,
                track_running_stats=args_.track_running_stats,
                shared_model=shared_model,
                args_=args_
            )

        logs_path = os.path.join(logs_root, "task_{}".format(task_id))
        os.makedirs(logs_path, exist_ok=True)
        logger = SummaryWriter(logs_path)

        client = get_client(
            client_type=CLIENT_TYPE[args_.method],
            learners_ensemble=learners_ensemble,
            q=args_.q,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=args_.local_steps,
            tune_locally=args_.locally_tune_clients,
            node_id=task_id,
            bi_level_opt=args_.bi_level_opt,
        )

        clients_.append(client)

    logging.info(f"total example numbers for train /val /test: {total_train_num} /{total_val_num}/ {total_test_num}")

    return clients_


def run_experiment(args_):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args_.seed)
    random.seed(args_.seed)
    np.random.seed(args_.seed)

    import multiprocessing
    # multiprocessing.set_start_method('spawn')
    # multiprocessing.set_start_method('forkserver')
    multiprocessing.set_start_method('fork')

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        args_.__setattr__("os_gpu_id", os.environ["CUDA_VISIBLE_DEVICES"])

    if args_.outdir == "":
        args_.outdir = os.path.join(os.getcwd(), "exp")
        args_.outdir = os.path.join(args_.outdir, args_.expname)

    if args_.model_type is not None:
        if "track_bn_0" in args_.model_type:
            args_.track_running_stats = 0
        if "track_bn_1" in args_.model_type:
            args_.track_running_stats = 1

    # if exist, make directory with given name and time
    if os.path.isdir(args_.outdir) and os.path.exists(args_.outdir):
        # args_.outdir = args_.outdir + datetime.now().strftime('_%m-%d_%H:%M:%S')
        args_.outdir = os.path.join(args_.outdir, "sub_exp" + datetime.now().strftime('_%m-%d_%H:%M:%S'))
        if os.path.exists(args_.outdir):
            args_.outdir = args_.outdir + "_dup"
    # if not, make directory with given name
    os.makedirs(args_.outdir)

    if args_.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")
    logger = logging.getLogger()
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(args_.outdir, 'exp_print.log'))
    fh.setLevel(logging.DEBUG)
    logger_formatter = logging.Formatter("%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    fh.setFormatter(logger_formatter)
    logger.addHandler(fh)
    sys.stderr = sys.stdout

    try:
        import wandb
        if "," in args_.expname:
            # format: "exp_group_name,job_type,detail_exp_name"
            res = args_.expname.split(",")
            if len(res) == 2:
                exp_name, job_type = res[0], res[1]
                group_name = None
            if len(res) == 3:
                exp_name, job_type, group_name = res[0], res[1], res[2]
        else:
            exp_name = args_.expname if args_.expname is not "" else None
            job_type = group_name = None
        wandb.init(project="pFedGate", entity="anonymous_research", config=args_,
                   group=group_name, job_type=job_type, name=exp_name, notes=args_.notes, reinit=True)
        # to capture all python source code files in the current directory and all subdirectories as an artifact
        not_log_dirs = ["data", "wandb", "Kmeans_Cython"]
        log_codes(wandb.run, outdir=args_.outdir, root='.', exclude_fn=lambda path: path in not_log_dirs)
    except ImportError:
        logging.warning("not found wandb, will not track wandb related results")

    data_dir = get_data_dir(args_.dataset_name)
    if "." in args_.dataset_name:
        args_.dataset_name = args_.dataset_name.split(".")[0]

    if "logs_root" in args_:
        logs_root = args_.logs_root
    else:
        logs_root = os.path.join("logs", args_to_string(args_))
    logging.info(f"logs_root is {logs_root}")
    logging.info(f"Data dir is {os.path.abspath(data_dir)}")

    if args_.lr_model == -1:
        logging.info(f"Will adopt the same learning rate for base model and gating layer as {args_.lr_gating}")
        args_.lr_model = args_.lr_gating

    print("==> Global learner initialization..")
    global_learners_ensemble = \
        get_learners_ensemble(
            n_learners=args_.n_learners,
            name=args_.dataset_name,
            device=args_.device,
            optimizer_name=args_.optimizer,
            scheduler_name=args_.lr_scheduler,
            initial_lr=[args_.lr_model, args_.lr_gating],
            input_dim=args_.input_dimension,
            output_dim=args_.output_dimension,
            n_rounds=args_.n_rounds,
            seed=args_.seed,
            mu=args_.mu,
            gated_learner=(args_.method == 'pFedGate'),
            alpha=args_.alpha,
            beta=args_.beta,
            sparse_factor=args_.sparse_factor,
            block_wise_prune=(args_.block_wise_prune == 1),
            fine_grained_block_split=args_.fine_grained_block_split,
            track_running_stats=args_.track_running_stats,
            args_=args_
        )

    if args_.online_aggregate == 1:
        shared_model = deepcopy(global_learners_ensemble[0].model)
    else:
        shared_model = None

    print("==> Clients initialization..")
    clients = init_clients(
        args_,
        root_path=os.path.join(data_dir, "train"),
        logs_root=os.path.join(logs_root, "train"),
        shared_model=shared_model
    )

    if args_.test_unseen_clients == 1:
        print("==> Unseen Clients initialization..")
        unseen_clients = init_clients(
            args_,
            root_path=os.path.join(data_dir, "test"),
            logs_root=os.path.join(logs_root, "test"),
            shared_model=shared_model
        )
    else:
        unseen_clients = []

    logs_path = os.path.join(logs_root, "train", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_train_logger = SummaryWriter(logs_path)

    logs_path = os.path.join(logs_root, "test", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_test_logger = SummaryWriter(logs_path)

    if args_.decentralized:
        aggregator_type = 'decentralized'
    else:
        aggregator_type = AGGREGATOR_TYPE[args_.method]

    aggregator = \
        get_aggregator(
            aggregator_type=aggregator_type,
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            lr_lambda=args_.lr_lambda,
            lr=[args_.lr_model, args_.lr_gating],
            q=args_.q,
            mu=args_.mu,
            communication_probability=args_.communication_probability,
            sampling_rate=args_.sampling_rate,
            log_freq=args_.log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=unseen_clients,
            verbose=args_.verbose,
            seed=args_.seed,
            aggregate_sampled_clients=args_.aggregate_sampled_clients,
            online_aggregate=args_.online_aggregate,
            outdir=args_.outdir
        )

    print("Training..")
    pbar = tqdm(total=args_.n_rounds)
    current_round = 0
    # Main loop over args_.n_rounds communication rounds
    while current_round <= args_.n_rounds:
        # mix() include the 1. broadcast;  2. update clients;  3. aggregate
        aggregator.mix()

        if aggregator.c_round != current_round:
            pbar.update(1)
            current_round = aggregator.c_round

    if "save_path" in args_:
        save_root = os.path.join(args_.save_path)

        os.makedirs(save_root, exist_ok=True)
        aggregator.save_state(save_root)


if __name__ == "__main__":

    args_ = parse_args()
    run_experiment(args_)
