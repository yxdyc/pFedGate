We provide some searched hyper-parameters that generally achieve good 
performance below. 

Note: to make a fair comparison with previous works, we used the 
SGD optimizer in our study. But we empirically found that pFedGate's 
optimization is more sensitive on harder datasets and harder models, w.r.t. 
the combination of `lr_gating`, `lr_model` and `lr_scheduler` (e.g., see the
[hpo results](hpo-cifar100-s05.png)). The values reported below may be slightly
different from our camera-ready version while should have similar performance.
For the scenarios with un-stable optimization, doing HPO more efficiently on 
small agent data or using a more advanced optimizer is promising, which we leave as future work.

```bash
YOUR_WORK_PATH=xxx/xx
cd ${YOUR_WORK_PATH}/pFedGate || exit

echo " run pFedGate with sparse-factor=1 on EMNIST dataset with full participation"
python3 run_experiment.py emnist.sample01 pFedGate --n_learners 1 --bz 128 --log_freq 5 --device cuda --seed 1234 --verbose 1 
--expname emnist01_pFedGate --n_rounds=400 --optimizer=sgd --sparse_factor=1 --block_wise_prune=1 --fine_grained_block_split=5 --lr_gating=0.1 --lr_model=0.1 --lr_scheduler=multi_step --bi_level_opt 0 --online_aggregate 0  --sparse_factor_scheduler=constant

echo " run pFedGate with sparse-factor=0.5 on EMNIST dataset with full participation"
python3 run_experiment.py emnist.sample01 pFedGate --n_learners 1 --bz 128 --log_freq 5 --device cuda --seed 1234 --verbose 1 --expname emnist01_pFedGate --n_rounds=400 --optimizer=sgd --sparse_factor=0.5 --block_wise_prune=1 --fine_grained_block_split=5 --lr_gating=0.1 --lr_model=0.1 --lr_scheduler=reduce_on_plateau --bi_level_opt 0 --online_aggregate 0  --sparse_factor_scheduler=constant

echo " run pFedGate with sparse-factor=0.25 on EMNIST dataset with full participation"
python3 run_experiment.py emnist.sample01 pFedGate --n_learners 1 --bz 128 --log_freq 5 --device cuda --seed 1234 --verbose 1 --expname emnist01_pFedGate --n_rounds=400 --optimizer=sgd --sparse_factor=0.25 --block_wise_prune=1 --fine_grained_block_split=5 --lr_gating=0.5 --lr_model=0.01 --lr_scheduler=reduce_on_plateau --bi_level_opt 0 --online_aggregate 0  --sparse_factor_scheduler=constant

echo " run pFedGate with sparse-factor=0.3 on EMNIST dataset with client sampling and sampling-rate=0.2"
python3 run_experiment.py emnist.sample01 pFedGate --n_learners 1 --bz 128 --log_freq 5 --device cuda --seed 1234 --verbose 1 --expname emnist01_pFedGate_s02 --n_rounds=2000 --optimizer=sgd --sparse_factor=0.3 
--block_wise_prune=1 --fine_grained_block_split=5 --lr_gating=0.1 --lr_model=0.1 --lr_scheduler=reduce_on_plateau --bi_level_opt 0 --online_aggregate 0  --sparse_factor_scheduler=constant --sampling_rate=0.2

echo " run pFedGate with sparse-factor=0.5 on FEMNIST dataset with full participation"
python3 run_experiment.py femnist.sample015 pFedGate --n_learners 1 --bz 128 --log_freq 5 --device cuda --seed 1234 --verbose 1 --expname emnist01_pFedGate --n_rounds=400 --optimizer=sgd --sparse_factor=0.5 --block_wise_prune=1 --fine_grained_block_split=5 --lr_gating=0.1 --lr_model=0.1 --lr_scheduler=reduce_on_plateau --bi_level_opt 0 --online_aggregate 0  --sparse_factor_scheduler=constant

echo " run pFedGate with sparse-factor=0.5 on FEMNIST dataset with 20% participation"  
python3 run_experiment.py femnist.sample015 pFedGate --n_learners 1 --bz 128 --log_freq 5 --device cuda --seed 1234 --verbose 1 --expname emnist01_pFedGate --n_rounds=2000 --optimizer=sgd --sparse_factor=0.5 --block_wise_prune=1 --fine_grained_block_split=5 --lr_gating=0.5 --lr_model=0.1 --lr_scheduler=multi_step --seperate_trans=1 --bi_level_opt 0 --sparse_factor_scheduler=constant

echo " run pFedGate with sparse-factor=0.3 on FEMNIST dataset with 20% participation"  
python3 run_experiment.py femnist.sample015 pFedGate --n_learners 1 --bz 128 --log_freq 5 --device cuda --seed 1234 --verbose 1 --expname emnist01_pFedGate --n_rounds=2000 --optimizer=sgd --sparse_factor=0.3 --block_wise_prune=1 --fine_grained_block_split=5 --lr_gating=0.5 --lr_model=0.1 --lr_scheduler=multi_step --seperate_trans=1 --bi_level_opt 0 --sparse_factor_scheduler=constant

echo " run pFedGate with sparse-factor=1 on CIFAR-10 dataset with full participation"  
python3 run_experiment.py cifar10 pFedGate --n_learners 1 --log_freq 5 --device cuda --seed 1234 --verbose 1 --bi_level_opt 0 --online_aggregate 1 --block_wise_prune=1 --bz=128 --fine_grained_block_split=5 --local_steps=1 --lr_gating=0.05 --lr_model=0.05 --lr_scheduler=multi_step  --n_rounds=400 --optimizer=sgd --sparse_factor=1 --sparse_factor_scheduler=constant

echo " run pFedGate with sparse-factor=0.5 on CIFAR-10 dataset with full participation"  
python3 run_experiment.py cifar10 pFedGate --n_learners 1 --log_freq 5 --device cuda --seed 1234 --verbose 1 --bi_level_opt 0 --online_aggregate 1 --block_wise_prune=1 --bz=128 --fine_grained_block_split=5 --local_steps=1 --lr_gating=0.05 --lr_model=0.05 --lr_scheduler=multi_step  --n_rounds=400 --optimizer=sgd --sparse_factor=0.5 --sparse_factor_scheduler=constant

echo " run pFedGate with sparse-factor=0.3 on CIFAR-10 dataset with full participation"  
python3 run_experiment.py cifar10 pFedGate --n_learners 1 --log_freq 5 --device cuda --seed 1234 --verbose 1 --bi_level_opt 0 --online_aggregate 1 --block_wise_prune=1 --bz=128 --fine_grained_block_split=5 --local_steps=1 --lr_gating=0.03 --lr_model=1.5 --lr_scheduler=multi_step  --n_rounds=400 --optimizer=sgd --sparse_factor=0.3 --sparse_factor_scheduler=constant

echo " run pFedGate with sparse-factor=1 on CIFAR-100 dataset with full participation"  
python3 run_experiment.py cifar100.clients50 pFedGate --n_learners 1 --log_freq 5 --device cuda --seed 1234 --verbose 1 --bi_level_opt 0 --online_aggregate 1 --block_wise_prune=1 --bz=128 --fine_grained_block_split=5 --local_steps=1 --lr_gating=0.5 --lr_model=0.1 --lr_scheduler=multi_step --model_type=lenet5_fc_wide --n_rounds=400 --optimizer=sgd --seperate_trans=1 --sparse_factor=1 --sparse_factor_scheduler=constant

echo " run pFedGate with sparse-factor=0.5 on CIFAR-100 dataset with full participation" python3 run_experiment.py cifar100.clients50 pFedGate  --n_learners 1 --log_freq 5 --device cuda --seed 1234 --verbose 1 --bi_level_opt 0 --online_aggregate 1 --block_wise_prune=1 --bz=128 --fine_grained_block_split=5 --local_steps=1 --lr_gating=1.5 --lr_model=0.03 --lr_scheduler=multi_step --model_type=lenet5_fc_wide --n_rounds=400 --optimizer=sgd --seperate_trans=1 --sparse_factor=0.5 --sparse_factor_scheduler=constant

echo " run pFedGate with sparse-factor=0.3 on CIFAR-100 dataset with full participation"  
python3 run_experiment.py cifar100.clients50 pFedGate --n_learners 1 --log_freq 5 --device cuda --seed 1234 --verbose 1 --bi_level_opt 0 --online_aggregate 1 --block_wise_prune=1 --bz=128 --fine_grained_block_split=5 --local_steps=1 --lr_gating=0.5 --lr_model=0.05 --lr_scheduler=multi_step --model_type=lenet5_fc_wide --n_rounds=400 --optimizer=sgd --seperate_trans=1 --sparse_factor=0.3 --sparse_factor_scheduler=constant

```


