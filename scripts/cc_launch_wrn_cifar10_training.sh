#!/bin/bash
# Launch Slurm jobs for WRN-28-2 training with UPS for all seeds and all
# levels of supervision on CIFAR-10.

seeds=(960146 663829 225659)

for seed in "${seeds[@]}"
do
    echo "Submitting jobs for seed: $seed..."
    sbatch cc_train_cifar10_250.sh $seed
    sleep 10
    sbatch cc_train_cifar10_500.sh $seed
    sleep 10
    sbatch cc_train_cifar10_1000.sh $seed
    sleep 10
    sbatch cc_train_cifar10_2000.sh $seed
    sleep 10
    sbatch cc_train_cifar10_4000.sh $seed
    sleep 10
done