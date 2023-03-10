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
        "--submit", help="whether to actually submit the jobs", action="store_true"
    )
    return parser.parse_args()


def submit_job(base_file, machine_file, extra_files, exp_file, dry_run):
    with open(exp_file) as fd:
        job_name = yaml.load(fd, Loader=yaml.FullLoader).pop("run_name")

    os.environ["BASE_CONFIG_FILE"] = base_file
    os.environ["MACHINE_CONFIG_FILE"] = machine_file
    os.environ["EXTRA_CONFIG_FILES"] = (
        " ".join(extra_files) if extra_files else exp_file
    )
    os.environ["EXP_CONFIG_FILE"] = exp_file

    command = [
        "sbatch",
        f"--output={log_dir}/%j-{job_name}.txt",
        f"--job-name={job_name}",
        # f"--export=BASE_CONFIG_FILE={base_file},MACHINE_CONFIG_FILE={machine_file},EXP_CONFIG_FILE={exp_file},EXTRA_CONFIG_FILES=",
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

    base_file = get_config_file(args.base)
    machine_file = get_config_file(args.machine)
    extra_files = [get_config_file(e) for e in args.extra]

    for exp_file in get_exp_files(args.exp, args.exp_dir):
        submit_job(base_file, machine_file, extra_files, exp_file, dry_run)


if __name__ == "__main__":
    main()
