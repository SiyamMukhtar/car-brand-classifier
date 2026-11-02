"""Train the car brand classifier (transfer learning with a ResNet).

Training happens in two phases:
    Phase 1  "head" : the pretrained layers are frozen, only the new final layer learns.
                      Fast and stable; it gives the new layer a sensible starting point.
    Phase 2  "fine-tune" : all layers are unfrozen and trained with a much smaller
                      learning rate, so the pretrained features adapt to car brands
                      without being destroyed.

After every epoch the model is scored on the VALIDATION set. The checkpoint with the best
validation macro-F1 is saved to models/<arch>_best.pth. The TEST set is never used here.

Examples:
    python src/train.py --subset 64 --head-epochs 1 --finetune-epochs 1 --batch-size 16   (quick check)
    python src/train.py --arch resnet18                                                   (full run)
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Draw to files only; no window is needed.
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, get_dataloaders
from model import SUPPORTED_ARCHS, build_model, set_backbone_trainable

ROOT = Path(__file__).resolve().parent.parent
SEED = 42


# Helper
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the car brand classifier.")
    p.add_argument("--arch", default="resnet18", choices=SUPPORTED_ARCHS)
    p.add_argument("--data-dir", default=str(ROOT / "data"),
                   help="Folder that contains clean/, splits.csv and class_names.json.")
    p.add_argument("--out-dir", default=str(ROOT),
                   help="models/ and outputs/ are created inside this folder.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--head-epochs", type=int, default=3, help="Epochs with the backbone frozen.")
    p.add_argument("--finetune-epochs", type=int, default=12, help="Epochs with all layers training.")
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=4,
                   help="Stop if validation macro-F1 does not improve for this many epochs.")
    p.add_argument("--workers", type=int, default=0 if os.name == "nt" else 2,
                   help="Data loading processes (0 on Windows).")
    p.add_argument("--subset", type=int, default=0,
                   help="Use only this many images from train and val (quick code check).")
    return p.parse_args()


def set_seed(seed: int) -> None:
    """Make runs repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def limit_loader(loader: DataLoader, n: int, shuffle: bool) -> DataLoader:
    """Return a loader that uses only n random images of the original loader."""
    indices = random.Random(SEED).sample(range(len(loader.dataset)), min(n, len(loader.dataset)))
    return DataLoader(Subset(loader.dataset, indices), batch_size=loader.batch_size,
                      shuffle=shuffle, num_workers=loader.num_workers)


def make_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    """AdamW over the parameters that are currently trainable."""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)


def run_epoch(model, loader, criterion, device, optimizer=None, desc=""):
    """One pass over a loader. Trains if an optimizer is given, otherwise only evaluates.

    Returns (average loss, accuracy, macro-F1).
    """
    training = optimizer is not None
    model.train(training)  # train mode while learning; eval mode (fixed behaviour) otherwise.
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.set_grad_enabled(training):  # No gradients needed when only evaluating.
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)             # Forward pass: scores for every class.
            loss = criterion(outputs, labels)   # How wrong were the scores?

            if training:
                optimizer.zero_grad()           # Clear old gradients.
                loss.backward()                 # Backpropagation: find how to change each weight.
                optimizer.step()                # Update the weights.

            total_loss += loss.item() * images.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    n = len(all_labels)
    return (total_loss / n,
            accuracy_score(all_labels, all_preds),
            f1_score(all_labels, all_preds, average="macro"))


def save_curves(history: list, path: Path) -> None:
    """Plot loss and validation scores per epoch."""
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], marker="o", label="validation")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax2.plot(epochs, [h["train_acc"] for h in history], marker="o", label="train accuracy")
    ax2.plot(epochs, [h["val_acc"] for h in history], marker="o", label="validation accuracy")
    ax2.plot(epochs, [h["val_f1"] for h in history], marker="o", label="validation macro-F1")
    ax2.set_title("Accuracy / F1")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# Main

def main() -> None:
    args = parse_args()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}   Architecture: {args.arch}")

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "outputs").mkdir(parents=True, exist_ok=True)
    class_names = json.loads((data_dir / "class_names.json").read_text(encoding="utf-8"))

    loaders, class_weights = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.workers,
        clean_dir=data_dir / "clean", splits_csv=data_dir / "splits.csv")
    if args.subset:
        loaders["train"] = limit_loader(loaders["train"], args.subset, shuffle=True)
        loaders["val"] = limit_loader(loaders["val"], args.subset, shuffle=False)
    print(f"Train images: {len(loaders['train'].dataset)}   Validation images: {len(loaders['val'].dataset)}")

    model = build_model(args.arch, n_classes=len(class_names)).to(device)
    # Class weights make mistakes on rare brands cost more (see compute_class_weights).
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    history, best_f1, epochs_without_improvement = [], -1.0, 0
    optimizer, scheduler = None, None
    total_epochs = args.head_epochs + args.finetune_epochs
    checkpoint_path = out_dir / "models" / f"{args.arch}_best.pth"

    for epoch in range(1, total_epochs + 1):
        # Switch phase at the right epoch.
        if epoch == args.head_epochs + 1:                      # Start of phase 2.
            set_backbone_trainable(model, True)
            optimizer = make_optimizer(model, args.finetune_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs)
            print("\n--- Phase 2: fine-tuning all layers ---")
        elif epoch == 1:                                       # Start of phase 1.
            set_backbone_trainable(model, False)
            optimizer = make_optimizer(model, args.head_lr)
            print("--- Phase 1: training only the new final layer ---")

        start = time.time()
        train_loss, train_acc, _ = run_epoch(model, loaders["train"], criterion, device,
                                             optimizer, desc=f"Epoch {epoch} train")
        val_loss, val_acc, val_f1 = run_epoch(model, loaders["val"], criterion, device,
                                              desc=f"Epoch {epoch} val")
        if scheduler is not None:
            scheduler.step()

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1})
        print(f"Epoch {epoch:>2}/{total_epochs}  train loss {train_loss:.3f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.3f} acc {val_acc:.3f} macro-F1 {val_f1:.3f}  ({time.time() - start:.0f}s)")

        # Keep the best model (by validation macro-F1).
        if val_f1 > best_f1:
            best_f1, epochs_without_improvement = val_f1, 0
            torch.save({
                "state_dict": model.state_dict(),
                "arch": args.arch,
                "class_names": class_names,
                "img_size": IMG_SIZE,
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
                "epoch": epoch,
                "val_macro_f1": val_f1,
            }, checkpoint_path)
            print(f"           saved new best model -> {checkpoint_path.name}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"No improvement for {args.patience} epochs. Stopping early.")
                break

    (out_dir / "outputs" / f"history_{args.arch}.json").write_text(json.dumps(history, indent=2))
    save_curves(history, out_dir / "outputs" / f"curves_{args.arch}.png")
    print(f"\nBest validation macro-F1: {best_f1:.3f}")
    print(f"Model:  {checkpoint_path}")
    print(f"Curves: {out_dir / 'outputs' / f'curves_{args.arch}.png'}")


if __name__ == "__main__":
    main()