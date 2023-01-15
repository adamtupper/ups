#!/bin/bash
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH --time=24:00:00  # TODO: Estimate how long this will take
#SBATCH --mail-user=$USER_EMAIL
#SBATCH --mail-type=ALL

module purge

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
    --split-txt "run1" \
    --arch "wideresnet" \
    --no-progress
