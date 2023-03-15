"""
Builds the tree distance matrix so that it's cached.
"""
import argparse
import pathlib

import hierarchy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Directory to calculate tree distances for. Should contain train/ and val/ directories.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = pathlib.Path(args.input)
    hierarchy.build_tree_dist_matrix(input_dir)


if __name__ == "__main__":
    main()
