"""
Calculate IoU between manually segmented and Python-segmented abalone lip images.

Usage:
    python calculate_iou.py /path/to/folder /path/to/output.csv
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import csv

ALPHA_THRESHOLD = 127
WHITE_DIST_THRESHOLD = 10
LIP_SUFFIX = "_lip"


def manual_mask(path: Path) -> np.ndarray:
    """Foreground = alpha > threshold."""
    img = Image.open(path).convert("RGBA")
    arr = np.array(img)
    if arr.shape[2] < 4:
        raise ValueError(f"{path.name}: expected an alpha channel, none found")
    return arr[:, :, 3] > ALPHA_THRESHOLD


def python_mask(path: Path) -> np.ndarray:
    """Foreground = pixel colour sufficiently different from pure white."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(int)
    dist = np.sqrt(((arr - 255) ** 2).sum(axis=2))
    return dist > WHITE_DIST_THRESHOLD


def compute_iou(manual_mask: np.ndarray, python_mask: np.ndarray) -> dict:
    """
    Manual mask is treated as ground truth.
    Precision = of pixels Python called lip, fraction that manual agrees are lip.
    Recall = of pixels manual called lip, fraction that Python also caught.
    """
    if manual_mask.shape != python_mask.shape:
        raise ValueError(f"Shape mismatch: {manual_mask.shape} vs {python_mask.shape}")

    intersection = np.logical_and(manual_mask, python_mask).sum()
    union = np.logical_or(manual_mask, python_mask).sum()
    manual_area = manual_mask.sum()
    python_area = python_mask.sum()

    iou = intersection / union if union > 0 else float("nan")
    dice = (2 * intersection) / (manual_area + python_area) if (manual_area + python_area) > 0 else float("nan")
    precision = intersection / python_area if python_area > 0 else float("nan")
    recall = intersection / manual_area if manual_area > 0 else float("nan")

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "intersection_px": int(intersection),
        "union_px": int(union),
        "manual_area_px": int(manual_area),
        "python_area_px": int(python_area),
    }


def find_pairs(folder: Path):
    """
    Match manual files (no suffix) to Python files (_lip suffix) by base filename.
    Returns list of (base_name, manual_path, python_path).
    Reports any unmatched files rather than silently skipping them.
    """
    all_pngs = sorted(folder.glob("*.png"))
    lip_files = {p.stem[: -len(LIP_SUFFIX)]: p for p in all_pngs if p.stem.endswith(LIP_SUFFIX)}
    manual_files = {p.stem: p for p in all_pngs if not p.stem.endswith(LIP_SUFFIX)}

    pairs = []
    for base, manual_path in manual_files.items():
        if base in lip_files:
            pairs.append((base, manual_path, lip_files[base]))

    unmatched_manual = sorted(set(manual_files) - set(lip_files))
    unmatched_lip = sorted(set(lip_files) - set(manual_files))

    return pairs, unmatched_manual, unmatched_lip


def main(folder_str: str, output_csv: str):
    folder = Path(folder_str)
    pairs, unmatched_manual, unmatched_lip = find_pairs(folder)

    if unmatched_manual:
        print(f"WARNING: {len(unmatched_manual)} manual file(s) with no matching _lip file:")
        for name in unmatched_manual:
            print(f"  - {name}.png")
    if unmatched_lip:
        print(f"WARNING: {len(unmatched_lip)} _lip file(s) with no matching manual file:")
        for name in unmatched_lip:
            print(f"  - {name}{LIP_SUFFIX}.png")

    print(f"\nMatched {len(pairs)} pair(s). Calculating IoU...\n")

    results = []
    for base, manual_path, python_path in pairs:
        try:
            m_mask = manual_mask(manual_path)
            p_mask = python_mask(python_path)
            stats = compute_iou(m_mask, p_mask)
            stats["base_name"] = base
            results.append(stats)
            print(f"{base}: IoU={stats['iou']:.4f}  Dice={stats['dice']:.4f}  "
                  f"Precision={stats['precision']:.4f}  Recall={stats['recall']:.4f}")
        except Exception as e:
            print(f"ERROR processing {base}: {e}")

    if not results:
        print("No results to write.")
        return

    ious = [r["iou"] for r in results]
    dices = [r["dice"] for r in results]
    precisions = [r["precision"] for r in results]
    recalls = [r["recall"] for r in results]

    print(f"\n{'Metric':<12}{'Mean':>8}{'Median':>8}{'Min':>8}{'Max':>8}{'Std':>8}")
    for name, vals in [("IoU", ious), ("Dice", dices), ("Precision", precisions), ("Recall", recalls)]:
        print(f"{name:<12}{np.mean(vals):>8.4f}{np.median(vals):>8.4f}{np.min(vals):>8.4f}{np.max(vals):>8.4f}{np.std(vals):>8.4f}")

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "base_name", "iou", "dice", "precision", "recall",
                "intersection_px", "union_px", "manual_area_px", "python_area_px",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\nResults written to {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python calculate_iou.py /path/to/folder /path/to/output.csv")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
