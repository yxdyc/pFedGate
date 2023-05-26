# [ICML'23] Efficient Personalized Federated Learning via Sparse Model-Adaptation
This repository contains implementation for an efficient 
personalized Federated Learning approach, pFedGate. 
It adaptively generates personalized sparse local models accounting for 
both the clients' heterogeneous data distributions and resource constraints.
See more details in our [paper](http://arxiv.org/abs/2305.02776).

>Federated Learning (FL) aims to train machine learning models for multiple clients without sharing their own private data. Due to the heterogeneity of clients' local data distribution, recent studies explore the personalized FL that learns and deploys distinct local models with the help of auxiliary global models. However, the clients can be heterogeneous in terms of not only local data distribution, but also their computation and communication resources. The capacity and efficiency of personalized models are restricted by the lowest-resource clients, leading to sub-optimal performance and limited practicality of personalized FL. To overcome these challenges, we propose a novel approach named pFedGate for efficient personalized FL by adaptively and efficiently learning sparse local models. With a lightweight trainable gating layer, pFedGate enables clients to reach their full potential in model capacity by generating different sparse models accounting for both the heterogeneous data distributions and resource constraints. Meanwhile, the computation and communication efficiency are both improved thanks to the adaptability between the model sparsity and clients' resources. Further, we theoretically show that the proposed pFedGate has superior complexity with guaranteed convergence and generalization error. Extensive experiments show that pFedGate achieves superior global accuracy, individual accuracy and efficiency simultaneously over state-of-the-art methods. We also demonstrate that pFedGate performs better than competitors in the novel clients participation and partial clients participation scenarios, and can learn meaningful sparse local models adapted to different data distributions.

We provide the core implementations of our method in `pFedGate` directory 
and `models` directory, and re-use data generation and some existing 
FL methods implementations from the FedEM 
([Marfoq et al. NeurIPS'21](https://github.com/omarfoq/FedEM)).
The hyper-parameter searching, training and evaluation scripts are in 
`scripts` directory.

## Requirements
```
cvxpy==1.1.17
matplotlib==3.1.0
networkx==2.5.1
numba==0.53.1
numpy==1.19.5
pandas==1.1.5
Pillow==9.0.1
scikit_learn==1.0.2
scipy==1.5.4
seaborn==0.11.2
tensorboard==2.5.0
torch==1.9.0
torchvision==0.10.0
tqdm==4.61.1
wandb==0.12.0
```

## Datasets
The code-generation scripts for `DATA_NAME=[EMNIST, FEMNIST, CIFAR10, CIFAR100]
` datasets and detailed instructions are in `data/$DATA_NAME$` directory.

## Running Experiments
We provide detailed scripts and hyper-parameter searching scripts in 
`scripts` directory. For example, 
- to run pFedGate with sparse-factor=0.5 on EMNIST dataset, use `run_experiment.py emnist.sample01 pFedGate --n_learners 1 --bz 128 --log_freq 5 --device cuda --seed 1234 --verbose 1 --expname emnist01_pFedGate --n_rounds=400 --optimizer=sgd --sparse_factor=0.5 --block_wise_prune=1 --fine_grained_block_split=5 --lr_gating=0.1 --lr_model=0.1 --lr_scheduler=reduce_on_plateau --bi_level_opt 0 --online_aggregate 0  --sparse_factor_scheduler=constant`
- to do hyper-parameters optimization with the help of [wandb-sweep](https://docs.wandb.ai/guides/sweeps), use
  ```bash
  # 1. install wandb and login to the wandb host, the sweep id will be printed out.
  # You can modify the yaml files freely to customize your hyper-parameter search space and search strategies
  wandb sweep scripts/sweep_hpo/emnist01_norm1_cnn_leaf.yaml
  # wandb: Creating sweep from: emnist01_norm1_cnn_leaf.yaml
  # wandb: Created sweep with ID: ay31mv5a
  # wandb: View sweep at: http://xx.xx.xx.xx:8080/your_sweep_name/pFedGate/sweeps/ay31mv5a
 
  
  # 2. on agent machine prepare your codes and data, and setup wandb 
  wandb login --host=http:xx.xx.xx.xx.8080
  # run the agent and the hpo results will be collected in your wandb pannel
  nohup wandb agent your_name/pFedGate/sweep_id & 
  ```


## License
This project adopts the Apache-2.0 License. Please kindly cite our paper (and the respective papers of the methods used) if our work is useful for you:
```
@inproceedings{chen2023pFedGate,
  title={Efficient Personalized Federated Learning via Sparse Model-Adaptation},
  author={Daoyuan Chen and Liuyi Yao and Dawei Gao and Bolin Ding and Yaliang Li},
  booktitle={International Conference on Machine Learning},
  year={2023},
}
