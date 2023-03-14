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


def generate_grid(search_space):
    """
    Will remove all keys from search space.
    """

    if not search_space:
        yield {}
        return

    key = next(iter(search_space))
    value = search_space.pop(key)

    for trial in generate_grid(search_space):
        for v in value["choices"]:
            yield {**trial, key: v}


def main():
    args = parse_args()

    sweep_config = utils.load_config(args.sweep)

    run_name = sweep_config.pop("run_name")

    search_space = to_search_space(OmegaConf.to_container(sweep_config))

    # Check if we can just do grid search.
    sampling_strategy = "grid"
    count = 1
    for key, value in search_space.items():
        if "choices" not in value:
            # have to use random sampling
            sampling_strategy = "random"
            break

        count *= len(value["choices"])

    # We can just do grid search
    if sampling_strategy == "grid" and count < args.count:
        print("Doing grid search.")
        trials = generate_grid(search_space)
    else:
        print("Doing quasi-random search.")
        trials = halton.generate_search(search_space, args.count)

    # Make the output directory for the generated configs
    output_dir = pathlib.Path(args.output) / f"sweep-{run_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, trial in enumerate(tqdm(trials)):
        config = OmegaConf.create(
            {
                "seed": i,
                # Dont' save any checkpoints for sweeps.
                # We will always train another model for later use.
                "save": {"interval": None, "wandb": False},
                "run_name": f"{run_name}-{i}",
            }
        )
        for key, value in trial.items():
            OmegaConf.update(config, key, value)

        # Write the file
        path = output_dir / f"{config.run_name}.yaml"
        with open(path, "w") as fd:
            OmegaConf.save(config, fd)


if __name__ == "__main__":
    main()
