"""Data cleaning for the car brand dataset.

Reads images from data/raw/<Brand>/ and writes a cleaned copy to data/clean/<Brand>/.
The raw folder is never modified.

Each image goes through these checks, in this order:
    1. Corrupt      -> the file cannot be opened as an image.
    2. Too small    -> the shortest side is below MIN_SIDE pixels.
    3. Duplicate    -> a (near) identical image was already kept for this brand.
Images that pass are standardised (RGB, longest side <= MAX_SIDE, JPEG) and saved.

Usage:
    python src/clean.py                # clean every brand folder in data/raw
    python src/clean.py Toyota BMW     # clean only the given brands
"""

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import imagehash
from PIL import Image
from tqdm import tqdm

# Configuration

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "clean"
REPORT_CSV = ROOT / "data" / "clean_report.csv"

MIN_SIDE = 150       # Drop images whose shortest side is smaller than this (pixels).
MAX_SIDE = 512       # Shrink images whose longest side is larger than this (pixels).
DUP_DISTANCE = 4     # Max Hamming distance between two hashes to call them duplicates.
JPEG_QUALITY = 92    # Quality of the cleaned JPEG files.


@dataclass
class CleanStats:
    """Counters describing what happened to the images of one brand."""
    brand: str
    total: int = 0
    corrupt: int = 0
    small: int = 0
    duplicate: int = 0
    kept: int = 0

# Helper functions

def load_rgb_image(path: Path) -> Optional[Image.Image]:
    """Open an image fully and convert it to RGB. Return None if the file is unreadable."""
    try:
        with Image.open(path) as img:
            img.load()                 # Force decoding now, so broken files fail here.
            return img.convert("RGB")  # Same colour format for every image (no CMYK/gray/alpha).
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


def is_duplicate(new_hash: imagehash.ImageHash, kept_hashes: list) -> bool:
    """Check whether new_hash is (almost) equal to the hash of any image already kept."""
    return any(new_hash - other <= DUP_DISTANCE for other in kept_hashes)


def save_clean_image(img: Image.Image, out_path: Path) -> None:
    """Shrink the image if it is large (keeping its aspect ratio) and save it as JPEG."""
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    img.save(out_path, "JPEG", quality=JPEG_QUALITY)

# Main cleaning logic

def clean_brand(brand_dir: Path) -> CleanStats:
    """Clean all images of one brand and return the statistics."""
    stats = CleanStats(brand=brand_dir.name)
    out_dir = CLEAN_DIR / brand_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Remove output of a previous run so re-running never mixes old and new files.
    for old_file in out_dir.glob("*.jpg"):
        old_file.unlink()

    kept_hashes = []  # Perceptual hashes of the images kept so far (for duplicate detection).

    for path in tqdm(sorted(brand_dir.iterdir()), desc=brand_dir.name):
        if not path.is_file():
            continue
        stats.total += 1

        # Step 1: corrupt or unreadable files.
        img = load_rgb_image(path)
        if img is None:
            stats.corrupt += 1
            continue

        # Step 2: images that are too small to carry useful information.
        if min(img.size) < MIN_SIDE:
            stats.small += 1
            continue

        # Step 3: duplicates. A perceptual hash is a small "fingerprint" of the picture,
        # so resized or re-compressed copies of the same photo get almost the same hash.
        img_hash = imagehash.phash(img)
        if is_duplicate(img_hash, kept_hashes):
            stats.duplicate += 1
            continue
        kept_hashes.append(img_hash)

        # Step 4: standardise and save.
        save_clean_image(img, out_dir / f"{brand_dir.name}_{stats.kept:04d}.jpg")
        stats.kept += 1

    return stats


def print_report(all_stats: list) -> None:
    """Print a summary table and the class-imbalance ratio."""
    print(f"\n{'class':16}{'total':>7}{'corrupt':>9}{'small':>7}{'dup':>6}{'kept':>7}")
    for s in all_stats:
        print(f"{s.brand:16}{s.total:>7}{s.corrupt:>9}{s.small:>7}{s.duplicate:>6}{s.kept:>7}")

    total_in = sum(s.total for s in all_stats)
    total_kept = sum(s.kept for s in all_stats)
    print(f"\nKept {total_kept} of {total_in} images ({total_in - total_kept} removed).")

    kept_counts = [s.kept for s in all_stats if s.kept > 0]
    if kept_counts:
        ratio = max(kept_counts) / min(kept_counts)
        print(f"Class imbalance (largest / smallest class): {ratio:.1f}x")


def save_report(all_stats: list) -> None:
    """Save the statistics to a CSV file so they can be quoted in the project report."""
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_stats[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(s) for s in all_stats)
    print(f"Report saved to {REPORT_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean the raw car images.")
    parser.add_argument("brands", nargs="*", help="Brand folders to clean (default: all).")
    args = parser.parse_args()

    if not RAW_DIR.exists():
        raise SystemExit(f"Raw data folder not found: {RAW_DIR}. Run download_dataset.py first.")

    if args.brands:
        brand_dirs = [RAW_DIR / name for name in args.brands]
        missing = [d.name for d in brand_dirs if not d.is_dir()]
        if missing:
            raise SystemExit(f"Brand folder(s) not found in data/raw: {', '.join(missing)}")
    else:
        brand_dirs = sorted(d for d in RAW_DIR.iterdir() if d.is_dir())

    all_stats = [clean_brand(d) for d in brand_dirs]
    print_report(all_stats)
    save_report(all_stats)


if __name__ == "__main__":
    main()