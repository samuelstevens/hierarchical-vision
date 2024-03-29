"""
To do simple shot classification, we simply put all of the images in the training set through the pretrained network to gather feature representations. Then we build a nice matrix of features (n_features x n_examples) and do nearest centroid classification. We include L2 normalization and centered L2-normalization options as well.
"""
import argparse
import os

import composer.functional
import numpy as np
import sklearn.metrics
import sklearn.neighbors
import torch
from composer.utils import dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

import algorithmic
import configs
import data
import hierarchy
import models
import utils
import wandb


def cache_path(config, is_train) -> str:
    save_dir = os.path.join(
        config.machine.save_root, "simpleshot-features", config.run_name
    )
    os.makedirs(save_dir, exist_ok=True)

    if is_train:
        filename = f"{config.train_dataset.path}-train-features.npy"
    else:
        filename = f"{config.eval_dataset.path}-eval-features.npy"

    return os.path.join(save_dir, filename)


def tree_distance(labels, preds, *, tree_dists):
    return np.sum(tree_dists[preds, labels]) / labels.size


def build_dataloader(config: configs.Config, is_train: bool) -> DataLoader:
    if is_train:
        split = "train"
        data_cfg = config.train_dataset
    else:
        split = "val"
        data_cfg = config.eval_dataset

    # Transforms
    transform = [transforms.ToTensor()]
    if data_cfg.resize_size > 0:
        transform.append(transforms.Resize(data_cfg.resize_size))
    transform.append(transforms.CenterCrop(data_cfg.crop_size))
    assert all(m < 1 for m in data_cfg.channel_mean)
    assert all(s < 1 for s in data_cfg.channel_std)
    transform.append(
        transforms.Normalize(mean=data_cfg.channel_mean, std=data_cfg.channel_std)
    )
    transform = transforms.Compose(transform)

    path = config.machine.datasets[data_cfg.path]
    if config.simpleshot.hierarchical:
        dataset = hierarchy.HierarchicalImageFolder(
            os.path.join(path, split), transform
        )
    else:
        dataset = data.ImageFolder(os.path.join(path, split), transform)

    sampler = dist.get_sampler(dataset, drop_last=False, shuffle=False)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=data_cfg.global_batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    return dataloader


# This is basically an anonymous function but python only has single-expression lambdas.
# I think it's clearer to put it in a named function.
def load_backbone(config, model):
    checkpoint = algorithmic.parse_checkpoint(config.model.pretrained_checkpoint)
    algorithmic.PretrainedBackbone.load_pretrained_backbone(
        checkpoint, model, config.machine.save_root, strict=False
    )


def load_features(config, is_train):
    dataloader = build_dataloader(config, is_train)

    # Load classes
    classes = []
    for _, outputs in tqdm(dataloader, desc="Getting classes"):
        classes.append(outputs)
    classes = torch.cat(classes, dim=0).numpy()

    cache = cache_path(config, is_train)
    if os.path.isfile(cache):
        print(f"Using cached features at {cache}.")
        return np.load(cache), classes

    # We are not going to use the final class predictions, so we can just use 2 classes.
    model = models.build_model(config, 2)

    functional_algorithms = {
        "BlurPool": composer.functional.apply_blurpool,
        "ChannelsLast": composer.functional.apply_channels_last,
        "PretrainedBackbone": lambda model: load_backbone(config, model),
    }

    for algorithm in config.algorithms:
        fn = functional_algorithms[algorithm.cls]
        fn(model)

    model = model.cuda()

    # Load features
    with torch.no_grad():
        features = []
        for inputs, _ in tqdm(dataloader, desc="Getting features"):
            inputs = inputs.cuda()
            features.append(model(inputs).cpu())
        features = torch.cat(features, dim=0).numpy()

    np.save(cache, features)

    return features, classes


def l2_normalize(features):
    # assume examples x features
    n_examples, n_features = features.shape
    assert n_features in (1024, 2048)

    norms = np.linalg.norm(features, ord=2, axis=1, keepdims=True)
    return features / norms


def center(features):
    # assume examples x features
    n_examples, n_features = features.shape
    assert n_features in (1024, 2048)

    mean = np.mean(features, axis=1, keepdims=True)
    return features / mean


def main(config):
    wandb.init(name=config.run_name)

    train_features, train_classes = load_features(config, is_train=True)
    print("Loaded train features.")
    test_features, test_classes = load_features(config, is_train=False)
    print("Loaded test features.")

    assert config.model.variant == "simpleshot", config.model.variant

    if config.simpleshot.centered:
        train_features = center(train_features)
        test_features = center(test_features)
    if config.simpleshot.l2_normalized:
        train_features = l2_normalize(train_features)
        test_features = l2_normalize(test_features)

    if config.simpleshot.hierarchical:
        lookups = hierarchy.build_parent_label_lookup(
            os.path.join(config.machine.datasets[config.train_dataset.path])
        )
        clf = hierarchy.HierarchicalNearestCentroid(lookups)
    else:
        clf = sklearn.neighbors.NearestCentroid()

    clf.fit(train_features, train_classes)

    preds = clf.predict(test_features)

    tree_dists = hierarchy.build_tree_dist_matrix(
        config.machine.datasets[config.eval_dataset.path]
    ).numpy()

    if config.simpleshot.hierarchical:
        # I probably could calculate some tier-level statistics (like kingdom accuracy)
        n_examples, n_tiers = test_classes.shape
        assert n_tiers == 7
        test_classes = test_classes[:, -1]

        n_preds, n_tiers = preds.shape
        assert n_tiers == 7
        preds = preds[:, -1]

    metrics = {
        "acc@1": np.sum(preds == test_classes) / len(test_classes),
        "tree-dist": tree_distance(test_classes, preds, tree_dists=tree_dists),
    }

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    wandb.log(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    utils.add_exp_args(parser)
    args = parser.parse_args()

    default_config = OmegaConf.structured(configs.Config)
    machine_config = utils.load_config(args.machine)
    exp_configs = [utils.load_config(file) for file in args.exp]

    config = OmegaConf.merge(
        default_config,
        machine_config,
        *exp_configs,
    )
    main(config)
