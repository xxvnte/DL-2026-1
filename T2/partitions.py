import json
import os

import torch
from torchvision import datasets

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
DATA = os.path.join(SCRIPT_DIR, "fish_image")

RANDOM_SEED = 42
SPLIT_TRAIN = 0.70
SPLIT_VAL = 0.15
SPLIT_TEST = 0.15


def resultPath(filename):
    return os.path.join(RESULTS_DIR, filename)


def main():
    torch.manual_seed(RANDOM_SEED)

    dataset = datasets.ImageFolder(root=DATA)
    n_images = len(dataset)

    n_train = int(SPLIT_TRAIN * n_images)
    n_val = int(SPLIT_VAL * n_images)
    n_test = n_images - n_train - n_val

    rng = torch.Generator().manual_seed(RANDOM_SEED)
    shuffled_idx = torch.randperm(n_images, generator=rng).tolist()

    train_idx = shuffled_idx[:n_train]
    val_idx = shuffled_idx[n_train : n_train + n_val]
    test_idx = shuffled_idx[n_train + n_val :]

    partitions = {
        "random_seed": RANDOM_SEED,
        "split_train": SPLIT_TRAIN,
        "split_val": SPLIT_VAL,
        "split_test": SPLIT_TEST,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = resultPath("partitions.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(partitions, f)

    print(f"Dataset: {n_images} images | {len(dataset.classes)} classes")
    print(f"Train: {n_train} | Val: {n_val} | Test: {n_test}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
