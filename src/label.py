"""Data labeling for the cleaned car brand dataset.

In this project the label of an image is simply the name of the folder it sits in
(data/clean/<Brand>/). This script turns that folder structure into explicit label files
and produces two figures for a visual sanity check.

Outputs:
    data/labels.csv           one row per image: filepath, brand, label_id
    data/class_names.json     list of brand names, position in the list = label_id
                              (the API will use it later to turn a predicted number back into a name)
    outputs/class_distribution.png   bar chart of images per brand
    outputs/sample_grid.png          a few random images of every brand

Usage:
    python src/label.py
"""

import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Draw to files only; no window is needed.
import matplotlib.pyplot as plt
from PIL import Image

# Configuration

ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT / "data" / "clean"
LABELS_CSV = ROOT / "data" / "labels.csv"
CLASS_NAMES_JSON = ROOT / "data" / "class_names.json"
OUTPUT_DIR = ROOT / "outputs"

IMAGES_PER_BRAND_IN_GRID = 6
SEED = 42  # Fixed seed so the sample grid is the same every time.

# Labeling

def scan_dataset():
    """Return (brand_folders, rows). Label ids follow the alphabetical order of the folders."""
    brand_folders = sorted(d.name for d in CLEAN_DIR.iterdir() if d.is_dir())
    rows = []
    for label_id, brand in enumerate(brand_folders):
        for path in sorted((CLEAN_DIR / brand).glob("*.jpg")):
            # Store paths relative to data/clean so the dataset can be moved (e.g. to Colab).
            rows.append((f"{brand}/{path.name}", brand, label_id))
    return brand_folders, rows


def write_label_files(brand_folders, rows):
    """Write labels.csv and class_names.json."""
    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "brand", "label_id"])
        writer.writerows(rows)

    display_names = [b.replace("_", " ") for b in brand_folders]  # "Aston_Martin" -> "Aston Martin"
    CLASS_NAMES_JSON.write_text(json.dumps(display_names, indent=2), encoding="utf-8")

# Figures

def plot_class_distribution(brand_folders, rows):
    """Horizontal bar chart of how many images each brand has."""
    counts = {b: 0 for b in brand_folders}
    for _, brand, _ in rows:
        counts[brand] += 1
    ordered = sorted(counts.items(), key=lambda item: item[1])  # Smallest at the bottom.

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(ordered) + 1.5))
    names = [b.replace("_", " ") for b, _ in ordered]
    values = [n for _, n in ordered]
    bars = ax.barh(names, values, color="#3b6ea5")
    ax.bar_label(bars, padding=3, fontsize=9)  # Print the count at the end of each bar.
    ax.set_xlabel("Number of images")
    ax.set_title(f"Images per brand (total {sum(values)})")
    ax.margins(x=0.1)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)


def plot_sample_grid(brand_folders, rows):
    """One row per brand with a few random images, to check labels with your own eyes."""
    rng = random.Random(SEED)
    n_cols = IMAGES_PER_BRAND_IN_GRID
    fig, axes = plt.subplots(len(brand_folders), n_cols,
                             figsize=(1.9 * n_cols, 1.4 * len(brand_folders)), squeeze=False)

    for r, brand in enumerate(brand_folders):
        files = [fp for fp, b, _ in rows if b == brand]
        picks = rng.sample(files, min(n_cols, len(files)))
        for c in range(n_cols):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(picks):
                with Image.open(CLEAN_DIR / picks[c]) as img:
                    ax.imshow(img)
            if c == 0:
                ax.set_title(brand.replace("_", " "), fontsize=9, loc="left")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sample_grid.png", dpi=110)
    plt.close(fig)


def main():
    if not CLEAN_DIR.exists():
        raise SystemExit(f"Clean data folder not found: {CLEAN_DIR}. Run clean.py first.")
    OUTPUT_DIR.mkdir(exist_ok=True)

    brand_folders, rows = scan_dataset()
    write_label_files(brand_folders, rows)
    plot_class_distribution(brand_folders, rows)
    plot_sample_grid(brand_folders, rows)

    print(f"Classes: {len(brand_folders)}   Images: {len(rows)}")
    for label_id, brand in enumerate(brand_folders):
        n = sum(1 for _, b, _ in rows if b == brand)
        print(f"  {label_id:>2}  {brand:15} {n}")
    print(f"\nSaved: {LABELS_CSV.name}, {CLASS_NAMES_JSON.name}, and 2 figures in outputs/")


if __name__ == "__main__":
    main()