import wandb
from omegaconf import DictConfig, OmegaConf


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
