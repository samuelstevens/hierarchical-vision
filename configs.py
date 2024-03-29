from dataclasses import dataclass, field
from typing import Any, Optional

Args = dict[str, Any]


@dataclass
class ModelConfig:
    name: str = "resnet50"
    # Can be full-tuning, linear-probing or simpleshot
    variant: str = "full-tuning"

    # If it exists, load this checkpoint from wandb.
    pretrained_checkpoint: Optional[str] = None


@dataclass
class DatasetConfig:
    # Must be a key in config.local.datasets
    path: str = ""
    # Image resize size before crop. -1 means no resize.
    resize_size: int = -1
    crop_size: int = 224
    global_batch_size: int = 2048

    drop_last: bool = False
    shuffle: bool = False

    # iNat21 training dataset defaults
    channel_mean: tuple[float, float, float] = (0.463, 0.480, 0.376)
    channel_std: tuple[float, float, float] = (0.238, 0.229, 0.247)


@dataclass
class MachineConfig:
    # Lookup from dataset name to dataset location.
    # Each dataset location has a train/ and val/ directory.
    datasets: dict[str, str] = field(default_factory=dict)
    # Where to save data.
    save_root: str = "."


@dataclass
class OptimConfig:
    name: str = "DecoupledSGDW"
    lr: float = 2.048
    momentum: float = 0.875
    weight_decay: float = 5e-4


@dataclass
class SchedulerConfig:
    name: str = "CosineAnnealingWithWarmupScheduler"
    args: Args = field(default_factory=lambda: {"t_warmup": "8ep", "alpha_f": 0.0})


# The only thing I'm ever doing is saving the last model or not saving anything.
# How can I simplify this configuration?
@dataclass
class SaveConfig:
    interval: Optional[str] = "10ep"
    num_checkpoints_to_keep: int = 1
    overwrite: bool = True
    # Whether to save checkpoints to a remote (wandb)
    wandb: bool = True


@dataclass
class WandbConfig:
    entity: str = "imageomics"
    project: str = "hierarchical-vision"


@dataclass
class SimpleShotConfig:
    centered: bool = False
    l2_normalized: bool = False
    hierarchical: bool = False


@dataclass
class AlgorithmConfig:
    cls: str = ""
    args: Args = field(default_factory=dict)


@dataclass
class HierarchyConfig:
    # Variant can be one of "multitask", "hxe" or empty ("") for disabled.
    variant: str = ""
    # Hierarchical coefficients for loss
    multitask_coeffs: list[float] = field(default_factory=list)
    # Weights of the levels of the tree. Can be "uniform" or "exponential"
    hxe_tree_weights: str = "uniform"
    # Factor for exponential weighting
    hxe_alpha: float = 0.1


@dataclass
class Config:
    run_name: str = "base"
    is_train: bool = True
    seed: int = 42
    max_duration: str = "90ep"
    grad_accum: str | int = "auto"
    load_path: Optional[str] = None
    # wandb tags
    tags: list[str] = field(default_factory=list)

    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)

    model: ModelConfig = field(default_factory=ModelConfig)

    train_dataset: DatasetConfig = field(default_factory=DatasetConfig)
    eval_dataset: DatasetConfig = field(default_factory=DatasetConfig)

    optim: OptimConfig = field(default_factory=OptimConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    algorithms: list[AlgorithmConfig] = field(default_factory=list)

    machine: MachineConfig = field(default_factory=MachineConfig)

    save: SaveConfig = field(default_factory=SaveConfig)

    wandb: WandbConfig = field(default_factory=WandbConfig)

    simpleshot: SimpleShotConfig = field(default_factory=SimpleShotConfig)
