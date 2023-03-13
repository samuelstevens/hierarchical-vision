import composer
from omegaconf import OmegaConf

import wandb


def log_config(config):
    # Only update the config on the master process
    # (which will have a wandb.run variable).
    if wandb.run:
        wandb.config.update(OmegaConf.to_container(config, resolve=True))
    print(OmegaConf.to_yaml(config))


def load_config(filepath: str):
    if not filepath:
        return OmegaConf.create()

    with open(filepath) as fd:
        return OmegaConf.load(fd)


def add_exp_args(parser):
    parser.add_argument(
        "--machine",
        help="Machine-specific YAML file (will include data folders, output folders, etc).",
        required=True,
    )
    parser.add_argument(
        "--exp",
        help="Experiment-specific YAML file (might have different a learning rate, for example). Will be applied left-to-right (right-most config has priority).",
        nargs="+",
        default=[],
        required=True,
    )


def save_last_only(state: composer.State, event: composer.Event):
    elapsed_duration = state.get_elapsed_duration()
    assert elapsed_duration is not None, "internal error"

    # Only checkpoint at end of training
    return elapsed_duration >= 1.0
