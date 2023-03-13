import argparse
import os

import einops
import torch
import torchvision
from tqdm.auto import tqdm


def load_statistics(directory, size=256):
    """
    Calculates mean and std for the individual channels so we can normalize images.
    """
    dataset = torchvision.datasets.ImageFolder(
        root=directory,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((size, size), antialias=True),
                torchvision.transforms.ToTensor(),
            ]
        ),
    )
    channels, height, width = dataset[0][0].shape
    assert channels == 3
    assert height == size
    assert width == size

    total = torch.zeros((channels,))
    total_squared = torch.zeros((channels,))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=os.cpu_count() * 12, num_workers=os.cpu_count()
    )

    for batch, _ in tqdm(dataloader):
        total += einops.reduce(batch, "batch channel height width -> channel", "sum")
        total_squared += einops.reduce(
            torch.mul(batch, batch), "batch channel height width -> channel", "sum"
        )

    divisor = len(dataset) * width * height

    mean = total / divisor
    var = total_squared / divisor - torch.mul(mean, mean)
    std = torch.sqrt(var)

    return mean, std


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, help="Directory with train/ and val/ folders."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mean, std = load_statistics(args.input)
    print(args.input, "mean:", mean.tolist(), "std:", std.tolist())


if __name__ == "__main__":
    main()
