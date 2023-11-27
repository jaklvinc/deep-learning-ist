#!/usr/bin/env python

import argparse
import os

from medmnist import OCTMNIST

def main():
    args = parse_args()

    root = args.root

    if not os.path.exists(root):
        os.makedirs(root)
    else:
        assert os.path.isdir(root), f"{root} is not a directory"

    _ = OCTMNIST(split="train", download=True, root=root)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".")
    return parser.parse_args()

if __name__ == "__main__":
    main()
