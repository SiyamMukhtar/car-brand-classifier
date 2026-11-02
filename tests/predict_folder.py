"""Send every image in a folder to the running API and print a results table.

The expected brand is read from the file name (e.g. 'bmw1.jpg' -> BMW). Files whose name
does not match one of the 20 brands (e.g. 'tesla1.jpg', 'cat1.jpg') are shown as
'not in the 20 brands'.

Usage (the API must be running):
    python tests/predict_folder.py
"""

import argparse
import csv
import json
import mimetypes
import re
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
NOT_IN_CLASSES = "not in the 20 brands"


def normalise(text: str) -> str:
    """Lower-case letters only, so 'Aston Martin', 'aston_martin' and 'astonmartin' match."""
    return re.sub(r"[^a-z]", "", text.lower())


def expected_brand(file_stem: str, class_names: list) -> str:
    """Find which brand a file name refers to (digits and separators are ignored)."""
    key = normalise(file_stem)
    for name in class_names:
        if normalise(name) == key:
            return name
    return NOT_IN_CLASSES


def main() -> None:
    p = argparse.ArgumentParser(description="Test the API with every image in a folder.")
    p.add_argument("--folder", default=str(ROOT / "test_images"))
    p.add_argument("--url", default="http://127.0.0.1:8000/predict")
    args = p.parse_args()

    class_names = json.loads((ROOT / "data" / "class_names.json").read_text(encoding="utf-8"))
    images = sorted(f for f in Path(args.folder).iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"No images found in {args.folder}")

    rows, correct, scored = [], 0, 0
    for path in images:
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        with open(path, "rb") as f:
            response = requests.post(args.url, files={"file": (path.name, f, mime)}, timeout=60)
        if response.status_code != 200:
            rows.append((path.name, "-", f"error {response.status_code}", "-", ""))
            continue
        result = response.json()
        if response.status_code != 200:
            rows.append((path.name, "-", f"error {response.status_code}", "-", ""))
            continue
        result = response.json()
        if not result.get("is_car", True):
            rows.append((path.name, expected_brand(re.sub(r"\d+$", "", path.stem), class_names),
                         "REJECTED (not a car)", "-", ""))
            continue
        truth = expected_brand(re.sub(r"\d+$", "", path.stem), class_names)
        truth = expected_brand(re.sub(r"\d+$", "", path.stem), class_names)
        verdict = ""
        if truth != NOT_IN_CLASSES:
            scored += 1
            ok = result["predicted_class"] == truth
            correct += ok
            verdict = "correct" if ok else "wrong"
        rows.append((path.name, truth, result["predicted_class"],
                     f"{result['confidence']:.1%}", verdict))

    # Print and save a Markdown table (it can be pasted straight into the README).
    lines = ["| File | Expected brand | Model prediction | Confidence | Result |",
             "|---|---|---|---|---|"]
    lines += [f"| {' | '.join(r)} |" for r in rows]
    if scored:
        lines.append(f"\nCorrect: {correct} of {scored} car photos ({correct / scored:.0%}).")
    table = "\n".join(lines)
    print(table)

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "api_test_results.md").write_text(table, encoding="utf-8")
    with open(out_dir / "api_test_results.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([("file", "expected", "predicted", "confidence", "result"), *rows])
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()