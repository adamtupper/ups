#!/bin/bash
# Launch Slurm jobs for CNN-13 training with UPS (no restarts) for all seeds and all
# levels of supervision on CIFAR-10.

seeds=(960146 663829 225659)

cd $project/ups

for seed in "${seeds[@]}"
do
    echo "Submitting jobs for seed: $seed..."
    sbatch scripts/cc_train_no_restarts_cnn13_cifar10_250.sh $seed
    sbatch scripts/cc_train_no_restarts_cnn13_cifar10_500.sh $seed
    sbatch scripts/cc_train_no_restarts_cnn13_cifar10_1000.sh $seed
    sbatch scripts/cc_train_no_restarts_cnn13_cifar10_2000.sh $seed
    sbatch scripts/cc_train_no_restarts_cnn13_cifar10_4000.sh $seed
done