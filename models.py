# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import composer.loss
import timm
import torch
import torchmetrics
from composer.metrics import CrossEntropy
from torchmetrics import Metric

import configs
import data
import hierarchy


def build_model(config: configs.Config, num_classes: int | list[int]):
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

    model.apply(weight_init)

    if config.model.variant == "linear-probing":
        # TODO: don't hardcode 2048
        model = LinearProbe(model, 2048, num_classes)
    elif config.model.variant == "full-tuning":
        pass
    elif config.model.variant in ("simpleshot", "simpleshot-l2n", "simpleshot-cl2n"):
        model = FeatureOnlyModel(model)
    else:
        raise ValueError(config.model.variant)

    return model


def build_composer_model(config: configs.Config, dataset_info: data.DatasetInfo):
    num_classes = dataset_info.num_classes

    model = build_model(config, num_classes)

    # Metrics
    # -------
    if config.hierarchy.variant == "multitask":
        train_metrics = {
            "cross-entropy": hierarchy.FineGrainedCrossEntropy(),
            "acc@1": hierarchy.FineGrainedAccuracy(topk=1),
            "acc@5": hierarchy.FineGrainedAccuracy(topk=5),
            "tree-dist": hierarchy.FineGrainedTreeDistance(
                tree_dists=dataset_info.tree_dists
            ),
        }
        val_metrics = {
            "cross-entropy": hierarchy.FineGrainedCrossEntropy(),
            "acc@1": hierarchy.FineGrainedAccuracy(topk=1),
            "acc@5": hierarchy.FineGrainedAccuracy(topk=5),
            "tree-dist": hierarchy.FineGrainedTreeDistance(
                tree_dists=dataset_info.tree_dists
            ),
        }
    else:
        assert isinstance(num_classes, int)

        train_metrics = {
            "cross-entropy": CrossEntropy(),
            "acc@1": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
            "acc@5": torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes, top_k=5
            ),
            "tree-dist": hierarchy.TreeDistance(tree_dists=dataset_info.tree_dists),
        }
        val_metrics = {
            "cross-entropy": CrossEntropy(),
            "acc@1": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
            "acc@5": torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes, top_k=5
            ),
            "tree-dist": hierarchy.TreeDistance(tree_dists=dataset_info.tree_dists),
        }

    # Loss Function
    # -------------
    if config.hierarchy.variant == "hxe":
        raise NotImplementedError()
    elif config.hierarchy.variant == "multitask":
        loss_fn = hierarchy.MultitaskCrossEntropy(
            coeffs=config.hierarchy.multitask_coeffs
        )
    elif config.hierarchy.variant == "":
        loss_fn = composer.loss.soft_cross_entropy
    else:
        raise ValueError(config.hierarchy.variant)

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


class LinearProbe(torch.nn.Module):
    def __init__(self, backbone, num_features, num_classes):
        super().__init__()

        self.backbone = backbone
        self.num_features = num_features
        self.num_classes = num_classes
        self.linear_layer = torch.nn.Linear(
            in_features=num_features, out_features=self.num_classes
        )

        self.backbone.eval()
        # Freeze the backbone
        self.backbone.requires_grad_(False)

    def forward(self, x):
        # Set to eval every step because we're not concerned with maximal throughput
        # and I don't want to do a forward pass with batchnorm or dropout in train mode.
        self.backbone.eval()

        # forward_features doesn't do any pooling.
        # forward_head with pre_logits=True doesn't do the resnet linear proj.
        features = self.backbone.forward_head(
            self.backbone.forward_features(x), pre_logits=True
        )  # B x Features

        logits = self.linear_layer(features)  # B x Classes

        return logits


class FeatureOnlyModel(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.backbone.eval()
        self.backbone.requires_grad_(False)

    def forward(self, x):
        # Set to eval every step because we're not concerned with maximal throughput
        # and I don't want to do a forward pass with batchnorm or dropout in train mode.
        self.backbone.eval()

        # forward_features doesn't do any pooling.
        # forward_head with pre_logits=True doesn't do the resnet linear proj.
        features = self.backbone.forward_head(
            self.backbone.forward_features(x), pre_logits=True
        )  # B x Features

        return features


def weight_init(w: torch.nn.Module):
    if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(w.weight)
    if isinstance(w, torch.nn.BatchNorm2d):
        w.weight.data = torch.rand(w.weight.data.shape)
        w.bias.data = torch.zeros_like(w.bias.data)
