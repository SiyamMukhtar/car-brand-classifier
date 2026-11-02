import csv
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
INFO_CSV = ROOT / "data" / "dataset_info.csv"

DATASET_ID = "tanganke/stanford_cars"
NUM_BRANDS = 20
MULTI_WORD_MAKES = ["AM General", "Aston Martin", "Land Rover"]


def get_make(class_name):
    """'Acura RL Sedan 2012' -> 'Acura', 'Land Rover LR2 SUV 2012' -> 'Land Rover'."""
    for make in MULTI_WORD_MAKES:
        if class_name.startswith(make + " "):
            return make
    return class_name.split()[0]


def main():
    RAW.mkdir(parents=True, exist_ok=True)

    parts = {}
    for split in ("train", "test"):
        print(f"Loading {split} split...")
        parts[split] = load_dataset(DATASET_ID, split=split)

    class_names = parts["train"].features["label"].names
    make_of = [get_make(n) for n in class_names]

    counts = Counter()
    for part in parts.values():
        for label in part["label"]:
            counts[make_of[label]] += 1

    print(f"Total makes found: {len(counts)}")
    chosen = {make for make, _ in counts.most_common(NUM_BRANDS)}
    print("Selected brands:")
    for make, n in counts.most_common(NUM_BRANDS):
        print(f"  {make}: {n} images")

    saved = Counter()
    with open(INFO_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "brand", "original_class", "original_split"])
        for split, part in parts.items():
            idxs = [i for i, label in enumerate(part["label"]) if make_of[label] in chosen]
            subset = part.select(idxs)
            for example in tqdm(subset, desc=f"Saving {split}"):
                make = make_of[example["label"]]
                folder_name = make.replace(" ", "_")
                folder = RAW / folder_name
                folder.mkdir(exist_ok=True)
                name = f"{folder_name}_{saved[make]:04d}.jpg"
                img = example["image"].convert("RGB")
                img.thumbnail((1024, 1024))
                img.save(folder / name, "JPEG", quality=92)
                writer.writerow([f"{folder_name}/{name}", make, class_names[example["label"]], split])
                saved[make] += 1

    print("Done. Images per brand:")
    for make, _ in counts.most_common(NUM_BRANDS):
        print(f"  {make}: {saved[make]}")


if __name__ == "__main__":
    main()