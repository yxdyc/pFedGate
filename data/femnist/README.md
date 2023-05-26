# FEMNIST Dataset

Most part of the data-generation codes are re-used from the FedEM
([Marfoq et al. NeurIPS'21](https://github.com/omarfoq/FedEM)).

## Introduction
This dataset is derived from the Leaf repository
([LEAF](https://github.com/TalwalkarLab/leaf)) pre-processing of the
Extended MNIST dataset, grouping examples by writer.

Details about LEAF were published in
"[LEAF: A Benchmark for Federated Settings"](https://arxiv.org/abs/1812.01097)

## Setup Instructions

First, run `./preprocess.sh`, then run `generate_data.py` with a choice of the following arguments:

- ```--s_frac```: fraction of the dataset to be used; default=``0.1``  
- ```--tr_frac```: train set proportion for each task; default=``0.8``
- ```--val_frac```: fraction of validation set (from train set); default=`0.0`
- ```--train_tasks_frac```: fraction of test tasks; default=``1.0``
- ```--seed``` : seed to be used before random sampling of data; default=``12345``

### Generation Scripts

In order to generate the used data split, run

```
python generate_data.py \
    --alpha 0.4 \
    --s_frac 0.15 \
    --tr_frac 0.8 \
    --seed 12345    
```

To include validation set, add `--val_frac 0.25`.