"""
Download CORD-v2 receipt dataset and convert to YOLO PII format.

Source  : naver-clova-ix/cord-v2 on HuggingFace
Content : 1000 real scanned receipt images with word-level quad bboxes
Covers  : PERSON_NAME (store name), ADDRESS (store address), PHONE, DATE_OF_BIRTH (date)

Annotation strategy:
  Each receipt comes with a ground_truth JSON containing a `valid_line` list.
  Each entry has a category label and per-word quadrilateral bboxes (absolute
  pixel coordinates). Words in the same line are merged into one annotation.
  Lines with the same PII category create separate annotations (address can
  span multiple lines). Non-PII categories (menu items, prices) are skipped.

Output mirrors data/funsd/ so prepare_dataset.py can merge all sources:
    <output>/
        images/  *.png
        labels/  *.txt   (YOLO format; empty file = background-negative)

Usage:
    python data/download_cord.py
    python data/download_cord.py --output data/cord
"""

import argparse
import json
import sys
from pathlib import Path

CLASSES = [
    "PERSON_NAME", "SSN", "ACCOUNT_NUMBER", "ADDRESS",
    "PHONE", "EMAIL", "DATE_OF_BIRTH", "SIGNATURE",
]
CID = {c: i for i, c in enumerate(CLASSES)}

# CORD category -> our PII class name  (all others are silently skipped)
CORD_CLASS_MAP = {
    "info.store_name": "PERSON_NAME",
    "info.store_addr": "ADDRESS",
    "info.phone":      "PHONE",
    "info.date":       "DATE_OF_BIRTH",
}


# ── bbox helpers ───────────────────────────────────────────────────────────────

def quad_to_bbox(quad: dict) -> list:
    """Convert a CORD quad dict (x1..x4, y1..y4) to [x1, y1, x2, y2]."""
    xs = [quad.get(f"x{i}", 0) for i in range(1, 5)]
    ys = [quad.get(f"y{i}", 0) for i in range(1, 5)]
    return [min(xs), min(ys), max(xs), max(ys)]


def merge_bbox(boxes: list) -> list:
    return [
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    ]


def bbox_to_yolo(bbox: list, img_w: int, img_h: int) -> tuple:
    """Convert absolute-pixel [x1,y1,x2,y2] -> YOLO (cx, cy, w, h) in [0,1]."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    return (
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.001, min(1.0, (x2 - x1) / img_w)),
        max(0.001, min(1.0, (y2 - y1) / img_h)),
    )


# ── ground-truth parser ────────────────────────────────────────────────────────

def parse_valid_lines(gt_json: str) -> list[tuple[str, list]]:
    """
    Parse a CORD ground_truth JSON string.
    Returns list of (category: str, merged_bbox: [x1,y1,x2,y2]).
    One entry per valid_line that maps to a PII class.
    """
    try:
        gt = json.loads(gt_json)
    except (json.JSONDecodeError, TypeError):
        return []

    results = []
    for line in gt.get("valid_line", []):
        category = line.get("category", "")
        cls_name = CORD_CLASS_MAP.get(category)
        if cls_name is None:
            continue

        word_bboxes = []
        for word in line.get("words", []):
            quad = word.get("quad")
            if quad:
                word_bboxes.append(quad_to_bbox(quad))

        if not word_bboxes:
            continue

        results.append((cls_name, merge_bbox(word_bboxes)))

    return results


# ── split converter ────────────────────────────────────────────────────────────

def convert_split(split_data, out_dir: Path, prefix: str) -> dict:
    imgs_dir   = out_dir / "images"
    labels_dir = out_dir / "labels"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "with_pii": 0, "annotations": 0, "by_class": {}}

    for idx, example in enumerate(split_data):
        img = example.get("image")
        gt_json = example.get("ground_truth", "{}")

        if img is None:
            continue

        img_w, img_h = img.size
        pii_lines = parse_valid_lines(gt_json)

        yolo_lines = []
        for cls_name, bbox in pii_lines:
            cx, cy, w, h = bbox_to_yolo(bbox, img_w, img_h)
            yolo_lines.append(f"{CID[cls_name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            stats["by_class"][cls_name] = stats["by_class"].get(cls_name, 0) + 1

        doc_id = f"{prefix}_{idx:04d}"
        img.convert("RGB").save(imgs_dir / f"{doc_id}.png")
        lbl_path = labels_dir / f"{doc_id}.txt"
        lbl_path.write_text("\n".join(yolo_lines), encoding="utf-8")

        stats["total"]       += 1
        stats["with_pii"]    += bool(yolo_lines)
        stats["annotations"] += len(yolo_lines)

        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  [{idx+1:4d}]  {doc_id}  {img_w}x{img_h}  pii_hits={len(yolo_lines)}", flush=True)

    return stats


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DocShield — download CORD-v2 -> YOLO PII format"
    )
    parser.add_argument("--output", default="data/cord")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' not installed.  Run: pip install datasets")
        sys.exit(1)

    out = Path(args.output)
    REPO = "naver-clova-ix/cord-v2"
    print(f"Downloading {REPO} from HuggingFace...")
    ds = load_dataset(REPO)

    totals = {"total": 0, "with_pii": 0, "annotations": 0, "by_class": {}}
    for split_name, prefix in [("train", "cord_tr"), ("validation", "cord_va"), ("test", "cord_te")]:
        if split_name not in ds:
            print(f"  Split '{split_name}' not in dataset, skipping.")
            continue
        print(f"\n-- {split_name.upper()} ({len(ds[split_name])} samples) --")
        stats = convert_split(ds[split_name], out, prefix)
        print(
            f"  {stats['total']} images | "
            f"{stats['with_pii']} with PII | "
            f"{stats['annotations']} annotations"
        )
        totals["total"]       += stats["total"]
        totals["with_pii"]    += stats["with_pii"]
        totals["annotations"] += stats["annotations"]
        for k, v in stats["by_class"].items():
            totals["by_class"][k] = totals["by_class"].get(k, 0) + v

    print(f"\nCORD done -> {out.resolve()}")
    print(f"  Total images      : {totals['total']}")
    print(f"  Images with PII   : {totals['with_pii']}")
    print(f"  Total annotations : {totals['annotations']}")
    for cls, cnt in sorted(totals["by_class"].items(), key=lambda x: -x[1]):
        print(f"    {cls}: {cnt}")
    print(f"\nNext: python data/prepare_dataset.py")


if __name__ == "__main__":
    main()
