import dataclasses
import re

import composer
import torch
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


@dataclasses.dataclass(frozen=True)
class WandbCheckpoint:
    source: str
    url: str
    filepath: str

    @classmethod
    def parse(cls, uri):
        match = re.match(r"^wandb://([\w./-]+:[\w./-]+)\?([\w./-]+)$", uri)
        if not match:
            raise ValueError(f"uri '{uri}' doesn't match pattern!")

        return cls("wandb", *match.groups())


def load_checkpoint_from_wandb(model, checkpoint, local_cache, strict=False):
    if isinstance(checkpoint, str):
        checkpoint = WandbCheckpoint.parse(checkpoint)
    elif isinstance(checkpoint, WandbCheckpoint):
        pass
    else:
        raise TypeError(checkpoint)

    downloaded_filepath = (
        wandb.run.use_artifact(checkpoint.url)
        .get_path(checkpoint.filepath)
        .download(root=local_cache)
    )
    state_dict = torch.load(downloaded_filepath, map_location="cpu")["state"]
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
        state_dict["model"], "module."
    )
    missing, unexpected = model.load_state_dict(state_dict["model"], strict=strict)
    if missing:
        print(f"Missing keys in checkpoint: {', '.join(missing)}")
    if unexpected:
        print(f"Unexpected keys in checkpoint: {', '.join(unexpected)}")
