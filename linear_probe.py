""""
Does linear classification with fixed features.

The features are feature representations from pre-trained deep learning
models, where each image produces 10 representations from torchvision's TenCrop.

1. Regular, center crop
2. Regular, random crop
3. Horizontal flip, center crop
4. Horizontal flip, random crop

For 10-shot 2000-way classification, we have 20K images x 4 "augmentations" giving us 
80K feature vectors with 2000 classes. Then we train a logistic regression classifier 
to predict the class from the feature vector.

We tune hyperparameter using k-fold classification. We use sklearn for simplicity.
"""

import argparse
import os

import composer.functional
import numpy as np
import sklearn.linear_model
import sklearn.metrics
import sklearn.pipeline
import torch
from composer.utils import dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, default_collate
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
        config.machine.save_root, "linear-probe-features", config.run_name
    )
    os.makedirs(save_dir, exist_ok=True)

    if is_train:
        filename = f"{config.train_dataset.path}-train-features.npy"
    else:
        filename = f"{config.eval_dataset.path}-eval-features.npy"

    return os.path.join(save_dir, filename)


def tree_distance(labels, preds, *, tree_dists):
    return np.sum(tree_dists[preds, labels]) / labels.size


def collate_tuples(samples):
    """
    samples: [(imgs, label)]
        where imgs is a tuple of images
    """
    inputs, outputs = [], []
    for imgs, label in samples:
        for img in imgs:
            inputs.append(img)
            outputs.append(label)

    return torch.stack(inputs), torch.tensor(outputs)


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
    assert all(m < 1 for m in data_cfg.channel_mean)
    assert all(s < 1 for s in data_cfg.channel_std)
    transform.append(
        transforms.Normalize(mean=data_cfg.channel_mean, std=data_cfg.channel_std)
    )
    # if is_train:
    # transform.append(transforms.TenCrop(data_cfg.crop_size))
    # collate_fn = collate_tuples
    # else:
    transform.append(transforms.CenterCrop(data_cfg.crop_size))
    collate_fn = default_collate

    transform = transforms.Compose(transform)

    path = config.machine.datasets[data_cfg.path]
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
        collate_fn=collate_fn,
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


def build_linear_model():
    return sklearn.model_selection.GridSearchCV(
        sklearn.pipeline.make_pipeline(
            sklearn.preprocessing.StandardScaler(),
            sklearn.linear_model.SGDClassifier(loss="log_loss"),
        ),
        {"sgdclassifier__alpha": [0.0001, 0.01, 1.0]},
        n_jobs=8,
        verbose=2,
    )


def main(config):
    wandb.init(name=config.run_name)

    train_features, train_classes = load_features(config, is_train=True)
    print("Loaded train features.")
    test_features, test_classes = load_features(config, is_train=False)
    print("Loaded test features.")

    # Shuffle the training data.
    random_indices = np.arange(len(train_features))
    np.random.default_rng().shuffle(random_indices)
    train_features = train_features[random_indices]
    train_classes = train_classes[random_indices]

    assert config.model.variant == "linear-probe", config.model.variant
    # Build the model
    clf = build_linear_model()
    print("Built the model.")

    # Fit the model.
    clf.fit(train_features, train_classes)
    print("Trained the model.")

    preds = clf.predict(test_features)
    print("Made predictions.")

    tree_dists = hierarchy.build_tree_dist_matrix(
        config.machine.datasets[config.eval_dataset.path]
    ).numpy()

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
