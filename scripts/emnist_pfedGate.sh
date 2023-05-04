YOUR_WORK_PATH=xxx/xx
cd ${YOUR_WORK_PATH}/pFedGate-Code-KDD || exit

echo " run pFedGate with sparse-factor=0.5 on EMNIST dataset with full participation"
python3 run_experiment.py emnist.sample01 pFedGate --n_learners 1 --bz 128 --log_freq 5 --device cuda --seed 1234 --verbose 1 --expname emnist01_pFedGate --n_rounds=400 --optimizer=sgd --sparse_factor=0.5 --block_wise_prune=1 --fine_grained_block_split=5 --lr_gating=0.1 --lr_model=0.1 --lr_scheduler=reduce_on_plateau --bi_level_opt 0 --online_aggregate 0  --sparse_factor_scheduler=constant


echo " run pFedGate with sparse-factor=0.3 on EMNIST dataset with client sampling and sampling-rate=0.2"
python3 run_experiment.py emnist.sample01 pFedGate --n_learners 1 --bz 128 --log_freq 5 --device cuda --seed 1234 --verbose 1 --expname emnist01_pFedGate --n_rounds=2000 --optimizer=sgd --sparse_factor=0.3 --block_wise_prune=1 --fine_grained_block_split=5 --lr_gating=0.1 --lr_model=0.1 --lr_scheduler=reduce_on_plateau --bi_level_opt 0 --online_aggregate 0  --sparse_factor_scheduler=constant --sampling_rate=0.2
