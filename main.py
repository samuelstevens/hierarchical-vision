# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""
Example script to train a ResNet model on ImageNet.

Further hacked on by Samuel Stevens
"""

import argparse
import os.path

import composer
import composer.algorithms
import composer.optim
import torch
from composer.callbacks import (
    CheckpointSaver,
    LRMonitor,
    MemoryMonitor,
    OptimizerMonitor,
    SpeedMonitor,
)
from composer.loggers import FileLogger
from composer.utils import dist, reproducibility
from omegaconf import OmegaConf

import configs
import data
import models
import monkey_patch
import optim
import utils


def main(config):
    reproducibility.seed_all(config.seed)
    if config.grad_accum == "auto" and not torch.cuda.is_available():
        raise ValueError(
            'grad_accum="auto" requires training with a GPU; please specify grad_accum as an integer'
        )

    # Divide batch sizes by number of devices if running multi-gpu training
    local_train_batch_size = config.train_dataset.global_batch_size
    local_eval_batch_size = config.eval_dataset.global_batch_size
    if dist.get_world_size():
        local_train_batch_size //= dist.get_world_size()
        local_eval_batch_size //= dist.get_world_size()

    # Train dataset
    print("Building train dataloader")
    train_dataspec, num_classes = data.build_dataspec(
        config, local_batch_size=local_train_batch_size, is_train=True
    )
    print("Built train dataloader\n")

    # Validation dataset
    print("Building evaluation dataloader")
    eval_dataspec, _ = data.build_dataspec(
        config, local_batch_size=local_eval_batch_size, is_train=False
    )
    print("Built evaluation dataloader\n")

    # Instantiate torchvision ResNet model
    print("Building Composer model")
    composer_model = models.build_composer_model(config, num_classes)
    print("Built Composer model\n")

    # Optimizer
    print("Building optimizer and learning rate scheduler")
    optimizer = optim.build_optimizer(config, composer_model)

    # Learning rate scheduler: LR warmup then cosine decay for the rest of training
    lr_scheduler = composer.optim.CosineAnnealingWithWarmupScheduler(
        t_warmup=config.scheduler.t_warmup, alpha_f=config.scheduler.alpha_f
    )
    print("Built optimizer and learning rate scheduler\n")

    # Callbacks for logging
    print("Building monitoring callbacks.")
    # Measures throughput as samples/sec and tracks total training time
    speed_monitor = SpeedMonitor(window_size=50)
    lr_monitor = LRMonitor()  # Logs the learning rate
    memory_monitor = MemoryMonitor()  # Logs memory utilization
    optim_monitor = OptimizerMonitor()

    # Callback for checkpointing
    print("Built monitoring callbacks\n")

    # Recipes for training ResNet architectures on ImageNet in order of increasing
    # training time and accuracy. To learn about individual methods, check out "Methods
    # Overview" in our documentation: https://docs.mosaicml.com/
    print("Building algorithm recipes")
    algorithms = []
    for algorithm in config.algorithms:
        cls = getattr(composer.algorithms, algorithm.cls)
        algorithms.append(cls(**algorithm.args))
    print("Built algorithm recipes\n")

    save_folder = os.path.join(config.save.root, config.run_name)
    checkpoint_saver = CheckpointSaver(
        folder=os.path.join(save_folder, "checkpoints"),
        overwrite=True,
        num_checkpoints_to_keep=config.save.num_checkpoints_to_keep,
        save_interval=config.save.interval,
    )
    loggers = [
        monkey_patch.WandBLogger(
            entity="imageomics",
            project="hierarchical-vision",
            log_artifacts=True,
            rank_zero_only=True,
            init_kwargs={"dir": save_folder},
        ),
        FileLogger(
            filename=os.path.join(save_folder, "logs", "log{rank}.txt"), overwrite=True
        ),
    ]

    # Create the Trainer!
    print("Building Trainer")
    device = "gpu" if torch.cuda.is_available() else "cpu"
    # Mixed precision for fast training when using a GPU
    precision = "amp" if device == "gpu" else "fp32"

    trainer = composer.Trainer(
        run_name=config.run_name,
        model=composer_model,
        train_dataloader=train_dataspec,
        eval_dataloader=eval_dataspec,
        eval_interval="1ep",
        optimizers=optimizer,
        schedulers=lr_scheduler,
        algorithms=algorithms,
        loggers=loggers,
        progress_bar=True,
        max_duration=config.max_duration,
        callbacks=[
            speed_monitor,
            lr_monitor,
            memory_monitor,
            optim_monitor,
            checkpoint_saver,
        ],
        save_folder=None,
        load_path=config.load_path,
        device=device,
        precision=precision,
        grad_accum=config.grad_accum,
        seed=config.seed,
    )
    print("Built Trainer\n")

    print("Logging config:\n")
    utils.log_config(config)

    print("Run evaluation")
    trainer.eval()
    if config.is_train:
        print("Train!")
        trainer.fit()


def parse_args():
    parser = argparse.ArgumentParser()
    utils.add_exp_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    default_config = OmegaConf.structured(configs.Config)
    base_config = utils.load_config(args.base)
    machine_config = utils.load_config(args.machine)
    extra_configs = [utils.load_config(file) for file in args.extra]
    exp_config = utils.load_config(args.exp)

    config = OmegaConf.merge(
        default_config,
        base_config,
        machine_config,
        *extra_configs,
        exp_config,
    )
    main(config)
