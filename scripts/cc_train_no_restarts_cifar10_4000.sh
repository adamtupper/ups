#!/bin/bash
#SBATCH --array=1-3%1
#SBATCH --mem=32000M
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH --time=3:00:00
#SBATCH --mail-user=adam.tupper.1@ulaval.ca
#SBATCH --mail-type=ALL

# Check for random seed
if [ -z "$1" ]; then
    echo "No seed supplied"
    exit 1
fi

# Check for (optional) experiment ID for resuming a previous training job
if [ -z "$2" ]; then
    $exp_id = $2
else
    $exp_id = "UPS_$SLURM_ARRAY_JOB_ID"
fi

# Print Job info
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
echo "Experiment ID: $exp_id"
echo ""
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."
echo ""

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

if test -d "$scratch/$exp_id"; then
    # Resume training run
    python train-cifar-without-restarts.py \
        --out $scratch \
        --data-dir $SLURM_TMPDIR/data \
        --resume "$scratch/$exp_id" \
        --exp-name $exp_id \
        --no-restarts \
        --dataset "cifar10" \
        --n-lbl 4000 \
        --seed $1 \
        --split-txt $exp_id \
        --arch "wideresnet" \
        --no-progress
else
    # Start a new training run
    python train-cifar-without-restarts.py \
        --out $scratch \
        --data-dir $SLURM_TMPDIR/data \
        --exp-name $exp_id \
        --no-restarts \
        --dataset "cifar10" \
        --n-lbl 4000 \
        --seed $1 \
        --split-txt $exp_id \
        --arch "wideresnet" \
        --no-progress
fi
