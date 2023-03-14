"""
To do simple shot classification, we simply put all of the images in the training set through the pretrained network to gather feature representations. Then we build a nice matrix of features (n_features x n_examples) and do nearest centroid classification. We include L2 normalization and centered L2-normalization options as well.
"""
import argparse
import os

import composer.functional
import torch
from omegaconf import OmegaConf

import configs
import data
import models
import monkey_patch
import utils
import wandb


def get_feature_vector(model, dataloader) -> torch.Tensor:
    breakpoint()
    # Need to find the number of examples in the entire dataspec
    features = torch.zeros(len(dataloader))
    for i, batch in dataloader:
        inputs, _ = batch
        features = model(inputs)


def main(config):
    wandb.init(name=config.run_name)

    train_dataspec, _ = data.build_dataspec(
        config, local_batch_size=config.train_dataset.global_batch_size, is_train=False
    )

    eval_dataspec, _ = data.build_dataspec(
        config, local_batch_size=config.eval_dataset.global_batch_size, is_train=False
    )

    # We are not going to use the final class predictions, so we can just use 2 classes.
    model = models.build_model(config, 2)
    # The only algorithms we need to apply are:
    # * BlurPool, because we trained with it.
    # * ChannelsLast because it's faster.
    # * PretrainedBackbone because we need the pretrained model.
    # Luckily they all have functional forms.
    # TODO: don't hardcode this.
    composer.functional.apply_blurpool(model)
    composer.functional.apply_channels_last(model)

    checkpoint = monkey_patch.WandbCheckpoint.parse(config.model.pretrained_checkpoint)
    monkey_patch.PretrainedBackbone.load_pretrained_backbone(
        checkpoint,
        model,
        os.path.join(config.machine.save_root, "wandb-artifacts", checkpoint.url),
        strict=False,
    )

    breakpoint()

    train_features = get_feature_vector(model, train_dataspec)
    test_features = get_feature_vector(model, eval_dataspec)

    # Need to do scikit learn nearest centroids
    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html
    raise NotImplementedError()


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
