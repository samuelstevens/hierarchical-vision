# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import torch
import torchvision.datasets
from composer.core import DataSpec
from composer.datasets.utils import NormalizationFn
from composer.utils import dist
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

import configs
import hierarchy

# Helpers
# -------
# Some helper classes/functions that are mostly wrappers/reimplementation of other code


class ImageFolder(torchvision.datasets.ImageFolder):
    """
    Adds a num_classes attribute to torchvisions.datasets.ImageFolder
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = len(self.classes)


def pil_image_collate(
    batch: list[tuple[Image.Image, Image.Image | np.ndarray]],
    memory_format: torch.memory_format = torch.contiguous_format,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reimplementation of composer.datasets.utils.pil_image_collate to handle targets with more than one dimension.
    """
    imgs = [sample[0] for sample in batch]
    w, h = imgs[0].size
    image_tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(
        memory_format=memory_format
    )

    # Convert targets to torch tensor
    targets = [sample[1] for sample in batch]
    if isinstance(targets[0], Image.Image):
        target_dims = (len(targets), targets[0].size[1], targets[0].size[0])
    # Added the next two lines
    elif isinstance(targets[0], torch.Tensor):
        target_dims = (len(targets), *targets[0].shape)
    else:
        target_dims = (len(targets),)
    target_tensor = torch.zeros(target_dims, dtype=torch.int64).contiguous(
        memory_format=memory_format
    )

    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)

        nump_array = np.rollaxis(nump_array, 2).copy()
        if nump_array.shape[0] != 3:
            assert nump_array.shape[0] == 1, "unexpected shape"
            nump_array = np.resize(nump_array, (3, h, w))
        assert image_tensor.shape[1:] == nump_array.shape, "shape mismatch"

        image_tensor[i] += torch.from_numpy(nump_array)
        target_tensor[i] += torch.from_numpy(np.array(targets[i], dtype=np.int64))

    return image_tensor, target_tensor


# Builder
# -------
# Main function for data.py


def build_dataspec(
    config: configs.Config,
    local_batch_size: int,
    is_train: bool = True,
    **dataloader_kwargs,
) -> tuple[DataSpec, int]:
    if is_train:
        split = "train"
        data_cfg = config.train_dataset
    else:
        split = "val"
        data_cfg = config.eval_dataset

    # Transforms
    # ----------
    transform = []
    if data_cfg.resize_size > 0:
        transform.append(transforms.Resize(data_cfg.resize_size))

    # split specific transformations
    if is_train:
        transform += [
            transforms.RandomResizedCrop(
                data_cfg.crop_size, scale=(0.08, 1.0), ratio=(0.75, 4.0 / 3.0)
            ),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        transform.append(transforms.CenterCrop(data_cfg.crop_size))

    transform = transforms.Compose(transform)

    # Scale by 255 since `pil_image_collate` results in images in range 0-255.
    # If we use ToTensor() and the default collate, remove the scaling
    if all(m < 1 for m in data_cfg.channel_mean):
        channel_mean = [m * 255 for m in data_cfg.channel_mean]
    if all(s < 1 for s in data_cfg.channel_std):
        channel_std = [s * 255 for s in data_cfg.channel_std]
    device_transform_fn = NormalizationFn(mean=channel_mean, std=channel_std)

    # Dataset
    # -------
    if config.hierarchy.variant == "multitask":
        dataset_cls = hierarchy.HierarchicalImageFolder
    else:
        dataset_cls = ImageFolder

    path = config.machine.datasets[data_cfg.path]
    dataset = dataset_cls(os.path.join(path, split), transform)
    sampler = dist.get_sampler(
        dataset, drop_last=data_cfg.drop_last, shuffle=data_cfg.shuffle
    )

    # DataSpec
    # --------
    # DataSpec allows for on-gpu transformations, slightly relieving dataloader bottleneck
    dataspec = DataSpec(
        DataLoader(
            dataset=dataset,
            batch_size=local_batch_size,
            sampler=sampler,
            drop_last=data_cfg.drop_last,
            **default_dataloader_kwargs,
            **dataloader_kwargs,
        ),
        device_transforms=device_transform_fn,
    )

    return dataspec, dataset.num_classes


default_dataloader_kwargs = dict(
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=pil_image_collate,
)
