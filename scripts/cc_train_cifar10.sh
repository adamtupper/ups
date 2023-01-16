#!/bin/bash
#SBATCH --array=831552,832388,727887,   # 3 random seeds
#SBATCH --mem=32000M
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH --time=24:00:00
#SBATCH --mail-user=adam.tupper.1@ulaval.ca
#SBATCH --mail-type=ALL

module purge

# Set Weights & Biases cache and output directories
export WANDB_CACHE_DIR=$SLURM_TMPDIR/.cache/wandb
export WANDB_DIR=$scratch/wandb
mkdir -p $WANDB_CACHE_DIR
mkdir -p $WANDB_DIR

# Copy data and code to compute node
mkdir $SLURM_TMPDIR/data
tar cf $project/ups.tar.gz $project/ups
tar xf $project/ups.tar.gz -C $SLURM_TMPDIR/ups
tar xf $project/data/cifar-10-python.tar.gz -C $SLURM_TMPDIR/data

cd $SLURM_TMPDIR/ups

# Create virtual environment
module load python/3.8.10 cuda cudnn
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r cc_requirements.txt

# Run training script
python train-cifar.py \
    --out $scratch \
    --data-dir $SLURM_TMPDIR/data \
    --dataset "cifar10" \
    --n-lbl 4000 \
    --seed $SLURM_ARRAY_TASK_ID \
    --split-txt "run$SLURM_ARRAY_TASK_ID" \
    --arch "wideresnet" \
    --no-progress
