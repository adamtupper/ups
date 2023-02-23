"""
python package_results.py
    --slurm-log-dir <path to slurm log directory>
    --artifacts-dir <path to output artifacts directory>
    --jobs <id1, id2, ...>
"""
import argparse
import glob
import os
import shutil
import tarfile


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--slurm-log-dir", type=str, required=True)
    parser.add_argument("--jobs", type=int, nargs="+", required=True)
    parser.add_argument("--artifacts-dir", type=str, required=True)

    return parser.parse_args()


def extract(line, prefix):
    """Extract metadata value given by `prefix` from the line if present."""
    if prefix in line:
        return line.split(prefix)[1].strip()
    else:
        return None


def process_lines(lines):
    """Process lines from Slurm log file to extract metadata."""
    experiment = None
    dataset = None
    num_labels = None
    seed = None
    arch = None
    iters = None
    epochs = None

    for line in lines:
        if experiment is None:
            experiment = extract(line, "Experiment ID:")
        if dataset is None:
            dataset = extract(line, "dataset:")
        if num_labels is None:
            num_labels = extract(line, "number of labeled samples:")
        if seed is None:
            seed = extract(line, "seed:")
        if arch is None:
            arch = extract(line, "architecture:")
        if iters is None:
            iters = extract(line, "number of pseudo-labeling iterations:")
        if epochs is None:
            epochs = extract(line, "number of epochs:")

    return experiment, dataset, num_labels, seed, arch, iters, epochs


def main():
    """Main function."""
    args = parse_args()

    for i, id in enumerate(args.jobs, start=1):
        print(f"Processing Job {id} ({i}/{len(args.jobs)})...")

        # Get list of Slurm log files
        slurm_logs = glob.glob(os.path.join(args.slurm_log_dir, f"slurm-{id}_*.out"))

        # Extract metadata from first the log file
        log_file = open(slurm_logs[0], "r")
        lines = log_file.readlines()
        experiment, dataset, num_labels, seed, arch, iters, epochs = process_lines(
            lines
        )

        # Move Slurm log files into experiment directory
        experiment_dir = os.path.join(args.artifacts_dir, f"{experiment}")
        for log_file in slurm_logs:
            shutil.copy2(log_file, experiment_dir)

        # Compile output files into compressed tarball
        tar_file = f"{dataset}_{num_labels}_{seed}_{arch}_iters{iters}_epochs{epochs}_exp{experiment}_job{id}.tar.gz"
        with tarfile.open(tar_file, "w:gz") as tar:
            tar.add(experiment_dir, arcname=os.path.basename(experiment_dir))

        # Clean-up
        shutil.rmtree(experiment_dir)
        for log_file in slurm_logs:
            os.remove(log_file)


if __name__ == "__main__":
    main()
