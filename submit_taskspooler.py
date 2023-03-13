"""
Script to submit taskspooler jobs.
"""

import argparse
import os
import subprocess

import utils


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

    # Every shell "word" has to be broken up into its own string, otherwise
    # subprocess.run will quote them and there will be errors.
    command = [
        "ts",
        "-G 1",
        "composer",
        "main.py",
        "--machine",
        machine_file,
        "--exp",
        *exp_files,
    ]

    if dry_run:
        print(" ".join(command))
        return

    try:
        output = subprocess.run(command, check=True, capture_output=True)
        print(output.stdout.decode("utf-8"), end="")
    except subprocess.CalledProcessError as e:
        print(e.stderr.decode("utf-8"), end="")
        print(e)


def get_exp_files(exp_dir):
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
