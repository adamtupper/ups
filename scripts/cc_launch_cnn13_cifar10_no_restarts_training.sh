#!/bin/bash
# Launch Slurm jobs for CNN-13 training with UPS (no restarts) for all seeds and all
# levels of supervision on CIFAR-10.

seeds=(960146 663829 225659)

for seed in "${seeds[@]}"
do
    echo "Submitting jobs for seed: $seed..."
    sbatch cc_train_no_restarts_cnn13_cifar10_250.sh $seed
    sleep 10
    sbatch cc_train_no_restarts_cnn13_cifar10_500.sh $seed
    sleep 10
    sbatch cc_no_restarts_train_cnn13_cifar10_1000.sh $seed
    sleep 10
    sbatch c_no_restartsc_train_cnn13_cifar10_2000.sh $seed
    sleep 10
    sbatch cc_no_restarts_train_cnn13_cifar10_4000.sh $seed
    sleep 10
done