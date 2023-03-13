"""
Makes a pretrain/downstream split by choosings some % of the species as "downstream".
"""
import argparse
import pathlib
import random
import shutil
import typing

from tqdm.auto import tqdm

import hierarchy

from . import concurrency, helpers

T = typing.TypeVar("T")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind",
        required=True,
        choices=["species", "genus", "subtree"],
        help="What kind of split to choose. 'species' chooses <frac> of species as a downstream task. 'genus' does the same at the genus level. 'subtree' tries to find a label in the tree that contains <total species> * <frac> species; so it might choose an entire Order with ~2000 species if frac is 0.2",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input directory. Should contain train/ and val/ directories.",
    )
    parser.add_argument(
        "--frac",
        required=True,
        type=float,
        help="Proportion of input classes to use as a downstream task. Must be between 0 and 1.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory. Will contain pretrain/ and downstream/ directories.",
    )
    # Optional arguments
    parser.add_argument("--seed", default=42, help="Random seed.")
    return parser.parse_args()


def get_classes(input_dir: pathlib.Path) -> set[str]:
    """
    Return a union of all classes in the train and validation set.
    """
    train_classes = {cls.stem for cls in (input_dir / "train").iterdir()}
    val_classes = {cls.stem for cls in (input_dir / "val").iterdir()}

    return train_classes | val_classes


def get_species(input_dir: pathlib.Path) -> set[str]:
    """
    Return a union of all species in the train and validation set.
    """
    train = {
        hierarchy.HierarchicalLabel.parse(cls.stem).species
        for cls in (input_dir / "train").iterdir()
    }
    val = {
        hierarchy.HierarchicalLabel.parse(cls.stem).species
        for cls in (input_dir / "val").iterdir()
    }

    return train | val


def species_match(cls: str, species: set[str]) -> bool:
    return hierarchy.HierarchicalLabel.parse(cls).species in species


def get_genuses(input_dir: pathlib.Path) -> set[str]:
    """
    Return a union of all genuses in the train and validation set.
    """
    train = {
        hierarchy.HierarchicalLabel.parse(cls.stem).genus
        for cls in (input_dir / "train").iterdir()
    }
    val = {
        hierarchy.HierarchicalLabel.parse(cls.stem).genus
        for cls in (input_dir / "val").iterdir()
    }

    return train | val


def genus_match(cls: str, genuses: set[str]) -> bool:
    return hierarchy.HierarchicalLabel.parse(cls).genus in genuses


def build_species_count_lookup(input_dir: pathlib.Path) -> hierarchy.LeafCountLookup:
    train_labels = {cls.stem for cls in (input_dir / "train").iterdir()}
    val_labels = {cls.stem for cls in (input_dir / "val").iterdir()}

    labels = sorted(train_labels | val_labels)

    return hierarchy.LeafCountLookup(
        [hierarchy.HierarchicalLabel.parse(label) for label in labels]
    )


def tier_match(cls: str, tier: str, label: str) -> bool:
    """
    Checks if the cls's tier matches the label.
    For example:
        cls: 00001_animalia_chordata_aves_accipitriformes
        tier: phylum
        label: chordata

    This matches because cls's phylum is chordata.
    """
    return getattr(hierarchy.HierarchicalLabel.parse(cls), tier) == label


def sample(classes: set[T], fraction: float, seed: int) -> set[T]:
    # round down to the nearest integer
    k = int(len(classes) * fraction)
    random.seed(seed)
    # Need to sort the classes first to convert from set (random order) to
    # list (fixed order) so that it reliably generates the same split.
    return set(random.sample(sorted(classes), k))


def copy_data(input_dir: pathlib.Path, output_dir: pathlib.Path, classes: set[str]):
    """
    input_dir has a train and a val directory.
    input_dir/<split> and input_dir/<split> both have class directories.
    input_dir/<split>/<CLS>/ has images.

    We add a train and val directory to output_dir, then mirror the structure of
    input_dir in output_dir. The only difference is that we only copy classes from
    input_dir that are in the classes argument.
    """

    # Helper function to get the output class
    # dir path for a given input class dir path.
    def output_dir_of(input_dir: pathlib.Path):
        *_, split, cls = input_dir.parts
        return output_dir / split / cls

    (output_dir / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "train").mkdir(parents=True, exist_ok=True)

    # We use a BoundedExecutor as a thin wrapper over
    # concurrent.futures.ThreadPoolExecutor, which lets us submit many "tasks"
    # to the operating system, which will run in parallel using different threads.
    # Using threads in this manner means we can copy data in parallel, which
    # massively speeds up copying, which is important for iNat21 (2.7M images)
    try:
        pool = concurrency.BoundedExecutor()

        def copy_data_split(split):
            """
            Copies all data for a given split (val, train).
            Made into a function so I don't duplicate it.
            """
            for class_dir in tqdm(sorted((input_dir / split).iterdir())):
                # Skip directories that aren't in classes
                if class_dir.name not in classes:
                    continue

                pool.submit(
                    shutil.copytree,
                    str(class_dir),
                    output_dir_of(class_dir),
                    dirs_exist_ok=False,
                )

        # Do validation first because it's faster.
        copy_data_split("val")
        copy_data_split("train")

        pool.finish(desc="Copying data")
    finally:
        pool.shutdown()


def main():
    args = parse_args()

    output_dir = pathlib.Path(args.output)
    logger = helpers.create_logger("rand-split", output_dir)

    input_dir = pathlib.Path(args.input)

    fraction = args.frac
    seed = args.seed

    all_classes = get_classes(input_dir)
    logger.info("Found %d input classes.", len(all_classes))

    if args.kind == "species":
        # 1. Get a list of all species
        all_species = get_species(input_dir)

        # 2. Randomly choose <fraction> of them as downstream
        downstream_species = sample(all_species, fraction, seed)

        # 3. Any class which has a downstream species is a downstream class.
        downstream_classes = {
            cls for cls in all_classes if species_match(cls, downstream_species)
        }
    elif args.kind == "genus":
        # 1. Get a list of all genuses
        all_genuses = get_genuses(input_dir)

        # 2. Randomly choose <fraction> of them as downstream
        downstream_genuses = sample(all_genuses, fraction, seed)

        # 3. Any class which has a downstream genus is a downstream class.
        downstream_classes = {
            cls for cls in all_classes if genus_match(cls, downstream_genuses)
        }
    elif args.kind == "subtree":
        # 1. Build up a lookup from tree label to number of species
        lookup = build_species_count_lookup(input_dir)
        # 2. Choose the node that is closest to <total species> * <frac>
        label, tier, count = lookup.closest(fraction)
        logger.info(
            "Picked a subtree. [label: %s, tier: %s, count: %d]", label, tier, count
        )
        # 3. Mark those species as downstream classes.
        downstream_classes = {
            cls for cls in all_classes if tier_match(cls, tier, label)
        }
    else:
        raise ValueError(args.kind)

    # 4. Other classes are for pretraining.
    pretrain_classes = all_classes - downstream_classes

    logger.info(
        "Split into pretrain/downstream. [pretrain: %d, downstream: %d]",
        len(pretrain_classes),
        len(downstream_classes),
    )
    logger.info("Last downstream class: %s", sorted(downstream_classes)[-1])
    logger.info("Last pretrain class: %s", sorted(pretrain_classes)[-1])

    # Do downstream first because it's faster.
    copy_data(input_dir, output_dir / "downstream", downstream_classes)
    copy_data(input_dir, output_dir / "pretrain", pretrain_classes)
    logger.info("Done.")


if __name__ == "__main__":
    main()
