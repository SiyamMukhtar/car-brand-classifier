"""Split the labeled dataset into train / validation / test sets.

Why three sets?
    train       : the model learns from these images.
    validation  : used during training to tune settings and detect overfitting.
    test        : touched only once at the very end, to report the final, honest result.

The split is STRATIFIED: every brand appears in train, validation and test in the same
proportion (70% / 15% / 15%), so small brands are not lost from a split by bad luck.

No image files are copied. The result is a single table, data/splits.csv, which adds a
"split" column to data/labels.csv. The training code reads this table.

Usage:
    python src/split.py
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Configuration

ROOT = Path(__file__).resolve().parent.parent
LABELS_CSV = ROOT / "data" / "labels.csv"
SPLITS_CSV = ROOT / "data" / "splits.csv"

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15      # The remaining 0.15 is the test set.
SEED = 42                # Fixed seed -> the same split every time (reproducible results).


def make_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with an extra 'split' column ('train', 'val' or 'test')."""
    # First cut: 70% train, 30% "rest".
    train_df, rest_df = train_test_split(
        df, train_size=TRAIN_FRACTION, stratify=df["label_id"], random_state=SEED
    )
    # Second cut: divide the "rest" equally into validation and test (15% + 15%).
    val_share_of_rest = VAL_FRACTION / (1 - TRAIN_FRACTION)
    val_df, test_df = train_test_split(
        rest_df, train_size=val_share_of_rest, stratify=rest_df["label_id"], random_state=SEED
    )

    out = pd.concat([
        train_df.assign(split="train"),
        val_df.assign(split="val"),
        test_df.assign(split="test"),
    ])
    return out.sort_values("filepath").reset_index(drop=True)


def check_splits(df: pd.DataFrame) -> None:
    """Sanity checks: no image in two sets, every brand in every set."""
    assert df["filepath"].is_unique, "An image appears in more than one split!"
    n_classes = df["label_id"].nunique()
    for split in ("train", "val", "test"):
        assert df[df["split"] == split]["label_id"].nunique() == n_classes, \
            f"Some brands are missing from the {split} set!"


def main() -> None:
    if not LABELS_CSV.exists():
        raise SystemExit(f"{LABELS_CSV} not found. Run label.py first.")

    df = pd.read_csv(LABELS_CSV)
    out = make_splits(df)
    check_splits(out)
    out.to_csv(SPLITS_CSV, index=False)

    # Report: images per brand in each split.
    table = out.pivot_table(index="brand", columns="split", values="filepath",
                            aggfunc="count")[["train", "val", "test"]]
    table["total"] = table.sum(axis=1)
    print(table.to_string())
    totals = out["split"].value_counts()
    print(f"\nTotal: train={totals['train']}  val={totals['val']}  test={totals['test']}"
          f"  (all={len(out)})")
    print("Checks passed: no overlap, every brand present in every split.")
    print(f"Saved: {SPLITS_CSV}")


if __name__ == "__main__":
    main()