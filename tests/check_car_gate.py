"""Measure how strongly an ImageNet model believes an image shows a car.

We add up the probabilities of all car-related ImageNet classes ("car probability").
Real car photos should score high, cats and food should score low. Run this script to see
the numbers on our own images, then pick a threshold between the two groups.

Usage:
    python tests/check_car_gate.py
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import models

ROOT = Path(__file__).resolve().parent.parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

# ImageNet class names that mean "a car or a part of a car".
CAR_CLASSES = {
    "ambulance", "beach wagon", "cab", "convertible", "jeep", "limousine", "minivan",
    "Model T", "pickup", "racer", "sports car", "police van", "minibus", "moving van",
    "tow truck", "car wheel", "grille",
}


def load_gate():
    """Pretrained ImageNet ResNet18 + the ids of the car-related classes."""
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights).eval()
    names = weights.meta["categories"]
    car_ids = [i for i, n in enumerate(names) if n in CAR_CLASSES]
    missing = CAR_CLASSES - {names[i] for i in car_ids}
    print(f"Car-related ImageNet classes found: {len(car_ids)} of {len(CAR_CLASSES)}"
          + (f"  (missing: {sorted(missing)})" if missing else ""))
    return model, weights.transforms(), car_ids


@torch.no_grad()
def car_probability(model, transform, car_ids, path: Path) -> float:
    image = Image.open(path).convert("RGB")
    probs = torch.softmax(model(transform(image).unsqueeze(0)), dim=1)[0]
    return float(probs[car_ids].sum())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--folder", default=str(ROOT / "test_images"))
    p.add_argument("--dataset-samples", type=int, default=150,
                   help="How many test-set car photos to score for comparison.")
    args = p.parse_args()

    model, transform, car_ids = load_gate()

    print("\n--- Your own photos (test_images) ---")
    for f in sorted(Path(args.folder).iterdir()):
        if f.suffix.lower() in IMAGE_SUFFIXES:
            print(f"{f.name:<22} car probability {car_probability(model, transform, car_ids, f):.3f}")

    print(f"\n--- {args.dataset_samples} random car photos from the test set ---")
    df = pd.read_csv(ROOT / "data" / "splits.csv")
    sample = df[df["split"] == "test"].sample(args.dataset_samples, random_state=0)
    scores = pd.Series([
        car_probability(model, transform, car_ids, ROOT / "data" / "clean" / fp)
        for fp in sample["filepath"]
    ])
    print(scores.describe(percentiles=[0.05, 0.10, 0.25, 0.5]).round(3).to_string())


if __name__ == "__main__":
    main()