"""
Does a qausi-random sweep over hyperparameters and generates config
files that should not be committed to version control.
"""
import argparse
import pathlib

from omegaconf import OmegaConf
from tqdm.auto import tqdm

import halton
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True, help="Config with sweep options.")
    parser.add_argument(
        "--count", type=int, default=50, help="Number of trials to sample."
    )
    parser.add_argument("--output", required=True, help="Output directory.")

    return parser.parse_args()


def to_search_space(dct: dict[str, object], sep="."):
    new = {}
    for key, value in dct.items():
        # Only flatten dicts that don't have (min, max, scaling) or (choices) key sets.
        if (
            isinstance(value, dict)
            and value.keys() != {"min", "max", "scaling"}
            and value.keys() != {"choices"}
        ):
            for nested_key, nested_value in to_search_space(value).items():
                new[key + sep + nested_key] = nested_value
            continue

        new[key] = value

    return new


def main():
    args = parse_args()

    sweep_config = utils.load_config(args.sweep)

    orig_run_name = sweep_config.pop("run_name")

    search_space = to_search_space(OmegaConf.to_container(sweep_config))

    # Make the output directory for the generated configs
    output_dir = pathlib.Path(args.output) / f"sweep-{orig_run_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, trial in enumerate(tqdm(halton.generate_search(search_space, args.count))):
        config = OmegaConf.create({"seed": i})
        for key, value in trial.items():
            OmegaConf.update(config, key, value)

        config.run_name = f"{orig_run_name}-{i}"

        # Write the file
        path = output_dir / f"{config.run_name}.yaml"
        with open(path, "w") as fd:
            OmegaConf.save(config, fd)


if __name__ == "__main__":
    main()
