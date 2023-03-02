#!/bin/bash
# Launch Slurm jobs for WRN-28-2 training with UPS for all seeds and all
# levels of supervision on CIFAR-10.

seeds=(960146 663829 225659)

cd $project/ups

for seed in "${seeds[@]}"
do
    echo "Submitting jobs for seed: $seed..."   
    sbatch scripts/cc_train_wrn_cifar10_1000.sh $seed
    sbatch scripts/cc_train_wrn_cifar10_4000.sh $seed
done