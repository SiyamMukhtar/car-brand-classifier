"""Evaluate the trained model on the TEST set (run this once, after training is finished).

Why a separate test set?
    The model never saw these images during training or model selection, so the score here
    is an honest estimate of how it will behave on new photos.

Outputs (in outputs/):
    test_report_<arch>.txt          accuracy, macro-F1, precision/recall/F1 per brand
    confusion_matrix_<arch>.png     which brands get mixed up with which
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from dataset import get_dataloaders
from model import build_model

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the model on the test set.")
    p.add_argument("--checkpoint", default=str(ROOT / "models" / "resnet18_best.pth"))
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--out-dir", default=str(ROOT / "outputs"))
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


@torch.no_grad()
def predict_all(model, loader, device):
    """Return true labels and predicted labels for every image in the loader."""
    model.eval()
    labels, preds = [], []
    for images, y in tqdm(loader, desc="Testing"):
        out = model(images.to(device))
        preds.extend(out.argmax(dim=1).cpu().tolist())
        labels.extend(y.tolist())
    return np.array(labels), np.array(preds)


def plot_confusion(cm: np.ndarray, class_names: list, path: Path) -> None:
    """Row-normalised confusion matrix: each row shows where that brand's photos went."""
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted brand")
    ax.set_ylabel("True brand")
    ax.set_title("Confusion matrix (test set, row-normalised)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def top_confusions(cm: np.ndarray, class_names: list, k: int = 8) -> list:
    """The k most frequent wrong (true -> predicted) pairs."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    order = np.dstack(np.unravel_index(np.argsort(-off, axis=None), off.shape))[0][:k]
    return [(class_names[i], class_names[j], int(off[i, j])) for i, j in order if off[i, j] > 0]


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # weights_only=False: the file also holds plain Python data (class names, mean/std).
    # It is our own file, so this is safe.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    class_names, arch = ckpt["class_names"], ckpt["arch"]

    model = build_model(arch, n_classes=len(class_names), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    data_dir = Path(args.data_dir)
    loaders, _ = get_dataloaders(batch_size=args.batch_size, num_workers=0,
                                 clean_dir=data_dir / "clean", splits_csv=data_dir / "splits.csv")
    y_true, y_pred = predict_all(model, loaders["test"], device)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    lines = [f"Architecture: {arch}   (best epoch {ckpt['epoch']}, val macro-F1 {ckpt['val_macro_f1']:.3f})",
             f"Test images : {len(y_true)}",
             f"Accuracy    : {acc:.3f}",
             f"Macro-F1    : {macro_f1:.3f}", "", report, "Most common mistakes (true -> predicted):"]
    lines += [f"  {t} -> {p}: {n} images" for t, p, n in top_confusions(cm, class_names)]
    text = "\n".join(lines)
    print(text)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"test_report_{arch}.txt").write_text(text, encoding="utf-8")
    plot_confusion(cm, class_names, out_dir / f"confusion_matrix_{arch}.png")
    print(f"\nSaved report and confusion matrix to {out_dir}")


if __name__ == "__main__":
    main()