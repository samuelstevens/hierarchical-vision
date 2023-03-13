"""
Script to submit slurm jobs.
"""

import argparse
import os
import subprocess

import yaml

import utils

log_dir = os.path.join(os.getcwd(), "logs")
job_file = os.path.join(os.getcwd(), "slurm", "train.bash")


def parse_args():
    parser = argparse.ArgumentParser()
    utils.add_exp_args(parser)
    parser.add_argument(
        "--exp-dir",
        help="Directory with configs. Submit a slurm job for each file in this directroy as the --exp option",
    )
    parser.add_argument(
        "--submit", help="Whether to actually submit the jobs", action="store_true"
    )
    parser.add_argument("--limit", help="How many jobs to submit.", default=0, type=int)
    return parser.parse_args()


def submit_job(machine_file, exp_files, dry_run):
    if not exp_files:
        raise ValueError("need at least one experiment file")

    with open(exp_files[-1]) as fd:
        job_name = yaml.load(fd, Loader=yaml.FullLoader).pop("run_name")

    environ = {
        "MACHINE_CONFIG_FILE": machine_file,
        "EXP_CONFIG_FILES": " ".join(exp_files),
    }

    for key, value in environ.items():
        os.environ[key] = value

    command = [
        "sbatch",
        f"--output={log_dir}/%j-{job_name}.txt",
        f"--job-name={job_name}",
        job_file,
    ]

    if dry_run:
        print(" ".join(command))
        return

    os.makedirs(log_dir, exist_ok=True)
    try:
        output = subprocess.run(command, check=True, capture_output=True)
        print(output.stdout.decode("utf-8"), end="")
    except subprocess.CalledProcessError as e:
        print(e.stderr.decode("utf-8"), end="")
        print(e)


def get_exp_files(exp, exp_dir):
    if exp:
        exp_file = os.path.join(os.getcwd(), exp)
        if os.path.isfile(exp_file):
            yield exp_file

    if exp_dir:
        exp_dir = os.path.join(os.getcwd(), exp_dir)
        if not os.path.isdir(exp_dir):
            print(f"'{exp_dir}' is not a directory!")
            return

        for entry in os.scandir(exp_dir):
            if entry.is_file():
                yield entry.path


def get_config_file(file):
    path = os.path.join(os.getcwd(), file)
    assert os.path.isfile(path), f"'{file}' does not exist!"
    return path


def main():
    args = parse_args()

    dry_run = not args.submit

    machine_file = get_config_file(args.machine)
    exp_files = [get_config_file(e) for e in args.exp]

    if args.exp_dir:
        submitted = 0
        for exp_file in get_exp_files(args.exp_dir):
            submit_job(machine_file, exp_files + [exp_file], dry_run)
            submitted += 1

            if args.limit > 0 and submitted >= args.limit:
                break
    else:
        submit_job(machine_file, exp_files, dry_run)


if __name__ == "__main__":
    main()
