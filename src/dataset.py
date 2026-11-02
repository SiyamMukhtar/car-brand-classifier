"""Preprocessing and data loading for training.

Preprocessing turns a stored JPEG into the exact tensor the CNN expects:
    1. Resize to 224 x 224 pixels        (the network needs a fixed input size)
    2. Convert to a tensor in [0, 1]     (a CNN works on numbers, not on JPEG files)
    3. Normalise with ImageNet mean/std  (the pretrained network was trained on data in this range)

Augmentation (random flip, small rotation, colour changes) is applied to TRAINING images only.
It shows the model a slightly different version of every photo in every epoch, which reduces
overfitting. Validation and test images are never augmented, so their scores stay fair.

Running this file prints a quick summary and saves outputs/train_batch_preview.png.

Usage:
    python src/dataset.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Draw to files only; no window is needed.
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Configuration

ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT / "data" / "clean"
SPLITS_CSV = ROOT / "data" / "splits.csv"
CLASS_NAMES_JSON = ROOT / "data" / "class_names.json"
OUTPUT_DIR = ROOT / "outputs"

IMG_SIZE = 224
# Mean / std of the ImageNet dataset. Pretrained torchvision models expect these values.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Transforms
def get_transforms(train: bool) -> transforms.Compose:
    """Return the preprocessing pipeline. Augmentation is added only when train=True."""
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    if train:
        return transforms.Compose([
            # Resize (not crop): a centre crop would cut off the front or rear of wide car photos.
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),   # A mirrored car is still the same brand.
            transforms.RandomRotation(10),            # Small tilt, like a hand-held photo.
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])

    # Validation / test / API: no randomness, so results are repeatable.
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

# Dataset and data loaders

class CarDataset(Dataset):
    """Loads one image and its label at a time, and applies the transforms."""

    def __init__(self, df: pd.DataFrame, clean_dir: Path, transform):
        self.paths = df["filepath"].tolist()
        self.labels = df["label_id"].tolist()
        self.clean_dir = Path(clean_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        with Image.open(self.clean_dir / self.paths[idx]) as img:
            img = img.convert("RGB")
        return self.transform(img), self.labels[idx]


def compute_class_weights(train_labels: pd.Series, n_classes: int) -> torch.Tensor:
    """Weight of a class = total / (n_classes * its count).

    Small classes get a weight above 1 and large classes below 1. Used in the loss function,
    so a mistake on a rare brand costs more and the model does not just favour big brands.
    """
    counts = train_labels.value_counts().sort_index()
    weights = len(train_labels) / (n_classes * counts)
    return torch.tensor(weights.values, dtype=torch.float32)


def get_dataloaders(batch_size: int = 32, num_workers: int = 0,
                    clean_dir: Path = CLEAN_DIR, splits_csv: Path = SPLITS_CSV):
    """Return ({'train','val','test': DataLoader}, class_weights).

    The paths can be changed, e.g. when the dataset is stored somewhere else on Google Colab.
    """
    df = pd.read_csv(splits_csv)
    loaders = {}
    for split in ("train", "val", "test"):
        part = df[df["split"] == split]
        dataset = CarDataset(part, clean_dir, get_transforms(train=(split == "train")))
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),  # Shuffle only the training data.
            num_workers=num_workers,
        )

    n_classes = df["label_id"].nunique()
    class_weights = compute_class_weights(df[df["split"] == "train"]["label_id"], n_classes)
    return loaders, class_weights

# Quick check
def denormalize(image: torch.Tensor) -> torch.Tensor:
    """Undo the normalisation so a tensor can be shown as a normal picture."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (image * std + mean).clamp(0, 1)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    class_names = json.loads(CLASS_NAMES_JSON.read_text(encoding="utf-8"))
    loaders, class_weights = get_dataloaders(batch_size=16)

    sizes = {name: len(loader.dataset) for name, loader in loaders.items()}
    print(f"Images -> train: {sizes['train']}  val: {sizes['val']}  test: {sizes['test']}")

    images, labels = next(iter(loaders["train"]))
    print(f"One batch: images {tuple(images.shape)}, labels {tuple(labels.shape)}")
    print(f"Pixel range after normalisation: {images.min():.2f} to {images.max():.2f}")

    low, high = class_weights.argmin().item(), class_weights.argmax().item()
    print(f"Class weight: lowest {class_weights[low]:.2f} ({class_names[low]}), "
          f"highest {class_weights[high]:.2f} ({class_names[high]})")

    # Save a 4 x 4 preview of augmented training images with their labels.
    fig, axes = plt.subplots(4, 4, figsize=(9, 9))
    for ax, image, label in zip(axes.flat, images, labels):
        ax.imshow(denormalize(image).permute(1, 2, 0))  # (C, H, W) -> (H, W, C) for matplotlib.
        ax.set_title(class_names[label], fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "train_batch_preview.png", dpi=110)
    plt.close(fig)
    print("Saved outputs/train_batch_preview.png")


if __name__ == "__main__":
    main()