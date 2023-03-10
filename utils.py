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
