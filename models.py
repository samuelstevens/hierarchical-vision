# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import composer.loss
import timm
import torch
import torchmetrics
from composer.metrics import CrossEntropy
from torchmetrics import Metric

import configs
import hierarchy


def build_composer_model(config: configs.Config, num_classes: int | list[int]):
    if isinstance(num_classes, int):
        # Simple flat baseline
        model = timm.create_model(config.model.name, num_classes=num_classes)
    elif isinstance(num_classes, tuple):
        assert (
            config.hierarchy.variant == "multitask"
        ), "config.hierarchy.variant must be multitask to use with multiple tiers of classes!"

        model = timm.create_model(config.model.name, num_classes=2)

        if hasattr(model, "fc"):
            hierarchy.multitask_surgery(model, "fc", num_classes)
        elif hasattr(model, "head"):
            hierarchy.multitask_surgery(model, "head", num_classes)
        else:
            raise NotImplementedError(
                "don't how to apply hierarchical multitask head to model!"
            )

    else:
        raise TypeError(
            f"num_classes must be int or (int, int ...), not {type(num_classes)}"
        )

    # Specify model initialization
    def weight_init(w: torch.nn.Module):
        if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(w.weight)
        if isinstance(w, torch.nn.BatchNorm2d):
            w.weight.data = torch.rand(w.weight.data.shape)
            w.bias.data = torch.zeros_like(w.bias.data)
        # When using binary cross entropy, set the classification layer bias to
        # -log(num_classes) to ensure the initial probabilities are approximately
        # 1 / num_classes
        if config.model.loss_name == "binary_cross_entropy" and isinstance(
            w, torch.nn.Linear
        ):
            w.bias.data = torch.ones(w.bias.shape) * -torch.log(
                torch.tensor(w.bias.shape[0])
            )

    model.apply(weight_init)

    # Metrics
    # -------
    if config.hierarchy.variant == "multitask":
        train_metrics = {"acc@1": hierarchy.FineGrainedAccuracy()}
        val_metrics = {
            "cross-entropy": hierarchy.FineGrainedCrossEntropy(),
            "acc@1": hierarchy.FineGrainedAccuracy(),
        }
    else:
        train_metrics = {
            "acc@1": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        }
        val_metrics = {
            "cross-entropy": CrossEntropy(),
            "acc@1": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
        }

    # Loss Function
    # -------------
    if config.hierarchy.variant == "hxe":
        raise NotImplementedError()
    elif config.hierarchy.variant == "multitask":
        loss_fn = hierarchy.MultitaskCrossEntropy(
            coeffs=config.hierarchy.multitask_coeffs
        )
    elif config.model.loss_name == "cross_entropy":
        loss_fn = composer.loss.soft_cross_entropy
    elif config.model.loss_name == "binary_cross_entropy":
        loss_fn = composer.loss.binary_cross_entropy_with_logits
    else:
        raise ValueError(config.model.loss_name, config.hierarchy.variant)

    return Model(
        model, train_metrics=train_metrics, val_metrics=val_metrics, loss_fn=loss_fn
    )


class Model(composer.ComposerModel):
    def __init__(
        self,
        module,
        train_metrics: dict[str, Metric],
        val_metrics: dict[str, Metric],
        loss_fn,
    ):
        super().__init__()
        self.module = module
        self.loss_fn = loss_fn

        self.train_metrics = train_metrics
        self.val_metrics = val_metrics

    def loss(self, outputs, batch, *args, **kwargs):
        _, targets = batch
        return self.loss_fn(outputs, targets, *args, **kwargs)

    def get_metrics(self, is_train=False):
        if is_train:
            return self.train_metrics
        else:
            return self.val_metrics

    def update_metric(self, batch, outputs, metric):
        _, targets = batch
        metric.update(outputs, targets)

    def forward(self, batch):
        inputs, _ = batch
        return self.module(inputs)