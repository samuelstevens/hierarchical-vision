import composer
from omegaconf import DictConfig, OmegaConf

import wandb


def log_config(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    if "wandb" in cfg.get("loggers", {}):
        if wandb.run:
            wandb.config.update(OmegaConf.to_container(cfg, resolve=True))


def load_config(filepath: str):
    if not filepath:
        return OmegaConf.create()

    with open(filepath) as fd:
        return OmegaConf.load(fd)


def add_exp_args(parser):
    parser.add_argument("--base", help="Base YAML file", required=True)
    parser.add_argument(
        "--machine",
        help="Machine-specific YAML file (will include data folders, output folders, etc).",
        required=True,
    )
    parser.add_argument(
        "--extra", help="Extra configs you want to include.", nargs="+", default=[]
    )
    parser.add_argument(
        "--exp",
        help="Experiment-specific YAML file (might have different a learning rate, for example).",
    )


def save_last_only(state: composer.State, event: composer.Event):
    elapsed_duration = state.get_elapsed_duration()
    assert elapsed_duration is not None, "internal error"

    # Only checkpoint at end of training
    return elapsed_duration >= 1.0


class LoadFromWandB(composer.Algorithm):
    def __init__(self, entity, project, checkpoint):
        # TODO: this checkpoint thing needs to be much clearer.
        # It interacts with the way wandb saves files.
        self.wandb_path = f"{entity}/{project}/{checkpoint}"
        self.filename = checkpoint

    def match(self, event, state):
        return event == composer.Event.INIT

    def apply(self, event, state, logger):
        if not wandb.run:
            raise RuntimeError("Call wandb.init before this!")

        downloaded_filepath = (
            wandb.run.use_artifact(self.path, type="model")
            .get_path(self.filename)
            .download()
        )
        composer.utils.load_checkpoint(
            downloaded_filepath, state, logger, load_weights_only=True
        )
