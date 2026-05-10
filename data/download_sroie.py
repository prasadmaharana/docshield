"""
Download SROIE-2019-v2 receipt dataset and convert to YOLO PII format.

Source  : rth/sroie-2019-v2 on HuggingFace
Content : 973 real scanned receipt images
Schema  : objects = {bbox: [[xs],[ys]], text: [...], entities: {company,date,address,total}}
Covers  : PERSON_NAME (company), ADDRESS, DATE_OF_BIRTH (date), PHONE (regex)

Annotation strategy:
  Ground-truth entity values (company / date / address) are matched against
  word-region text to locate the correct bboxes. Matching is word-overlap based
  to handle minor OCR drift. PHONE is detected via regex since SROIE entities
  does not include a phone field. TOTAL is not PII and is skipped.

Usage:
    python data/download_sroie.py
    python data/download_sroie.py --output data/sroie
"""

import argparse
import re
import sys
from pathlib import Path

CLASSES = [
    "PERSON_NAME", "SSN", "ACCOUNT_NUMBER", "ADDRESS",
    "PHONE", "EMAIL", "DATE_OF_BIRTH", "SIGNATURE",
]
CID = {c: i for i, c in enumerate(CLASSES)}

# SROIE entity key -> our PII class
ENTITY_CLASS_MAP = {
    "company": "PERSON_NAME",
    "date":    "DATE_OF_BIRTH",
    "address": "ADDRESS",
    # "total"  -> not PII, skip
}

_PHONE_RE = re.compile(
    r'\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    r'|\b\+?6?0\d[-.\s]?\d{7,8}\b'      # Malaysian format
)
_NORM_RE = re.compile(r'[^a-z0-9\s]')


# ── bbox helpers ───────────────────────────────────────────────────────────────

def normalise_bbox(raw) -> list[int]:
    """
    Convert SROIE bbox [[x1,x2,x3,x4], [y1,y2,y3,y4]] -> [x1,y1,x2,y2].
    Also handles flat [x1,y1,x2,y2] or quad [x1,y1,...,x4,y4] as fallback.
    """
    if raw and isinstance(raw[0], (list, tuple)):
        xs, ys = raw[0], raw[1]
        return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
    r = [int(v) for v in raw]
    if len(r) == 8:
        return [min(r[0::2]), min(r[1::2]), max(r[0::2]), max(r[1::2])]
    if len(r) == 4:
        x1, y1, x2, y2 = r
        return [x1, y1, x1 + x2, y1 + y2] if x2 <= x1 or y2 <= y1 else r
    return r[:4]


def merge_bbox(boxes: list) -> list:
    return [
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    ]


def bbox_to_yolo(bbox: list, img_w: int, img_h: int) -> tuple:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    return (
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.001, min(1.0, (x2 - x1) / img_w)),
        max(0.001, min(1.0, (y2 - y1) / img_h)),
    )


# ── entity -> bbox matching ─────────────────────────────────────────────────────

def _word_set(text: str) -> set:
    return set(_NORM_RE.sub(' ', text.lower()).split())


def find_entity_bboxes(texts: list, bboxes: list, entity_val: str) -> list:
    """
    Return bboxes of word regions whose text overlaps significantly with entity_val.
    Uses normalised word-set intersection so minor OCR drift doesn't break matching.
    """
    entity_words = _word_set(entity_val)
    if not entity_words:
        return []

    matched = []
    for text, bbox in zip(texts, bboxes):
        region_words = _word_set(text)
        if not region_words:
            continue
        overlap = len(region_words & entity_words)
        # Match if ≥40% of the region's words appear in the entity value
        if overlap > 0 and overlap / len(region_words) >= 0.4:
            matched.append(bbox)

    return matched


# ── split converter ────────────────────────────────────────────────────────────

def convert_split(split_data, out_dir: Path, prefix: str) -> dict:
    imgs_dir   = out_dir / "images"
    labels_dir = out_dir / "labels"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "with_pii": 0, "annotations": 0, "by_class": {}}

    for idx, example in enumerate(split_data):
        img = example.get("image") or example.get("img")
        if img is None:
            continue

        img_w, img_h = img.size
        objects  = example.get("objects", {})
        texts    = objects.get("text", [])
        raw_bbs  = objects.get("bbox", [])
        entities = objects.get("entities", {})

        bboxes = [normalise_bbox(b) for b in raw_bbs]

        yolo_lines = []

        # ── entity-matched annotations (company / date / address) ──────────
        for entity_key, cls_name in ENTITY_CLASS_MAP.items():
            entity_val = (entities.get(entity_key) or "").strip()
            if not entity_val:
                continue
            matched = find_entity_bboxes(texts, bboxes, entity_val)
            if not matched:
                continue
            cx, cy, w, h = bbox_to_yolo(merge_bbox(matched), img_w, img_h)
            yolo_lines.append(f"{CID[cls_name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            stats["by_class"][cls_name] = stats["by_class"].get(cls_name, 0) + 1

        # ── phone via regex (not in SROIE entities) ─────────────────────────
        for text, bbox in zip(texts, bboxes):
            if _PHONE_RE.search(text):
                cx, cy, w, h = bbox_to_yolo(bbox, img_w, img_h)
                yolo_lines.append(f"{CID['PHONE']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                stats["by_class"]["PHONE"] = stats["by_class"].get("PHONE", 0) + 1

        doc_id = f"{prefix}_{idx:04d}"
        img.convert("RGB").save(imgs_dir / f"{doc_id}.png")
        (labels_dir / f"{doc_id}.txt").write_text("\n".join(yolo_lines), encoding="utf-8")

        stats["total"]       += 1
        stats["with_pii"]    += bool(yolo_lines)
        stats["annotations"] += len(yolo_lines)

        if (idx + 1) % 100 == 0 or idx == 0:
            print(
                f"  [{idx+1:4d}]  {doc_id}  {img_w}x{img_h}  pii_hits={len(yolo_lines)}",
                flush=True,
            )

    return stats


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DocShield — download SROIE-2019-v2 -> YOLO PII format"
    )
    parser.add_argument("--output", default="data/sroie")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' not installed.  Run: pip install datasets")
        sys.exit(1)

    out  = Path(args.output)
    REPO = "rth/sroie-2019-v2"
    print(f"Downloading {REPO} from HuggingFace...")
    ds = load_dataset(REPO)

    totals = {"total": 0, "with_pii": 0, "annotations": 0, "by_class": {}}
    for split_name, prefix in [("train", "sroie_tr"), ("test", "sroie_te")]:
        if split_name not in ds:
            print(f"  Split '{split_name}' not found — skipping.")
            continue
        print(f"\n-- {split_name.upper()} ({len(ds[split_name])} samples) --")
        stats = convert_split(ds[split_name], out, prefix)
        print(
            f"  {stats['total']} images | "
            f"{stats['with_pii']} with PII | "
            f"{stats['annotations']} annotations"
        )
        for k, v in stats["by_class"].items():
            totals["by_class"][k] = totals["by_class"].get(k, 0) + v
        totals["total"]       += stats["total"]
        totals["with_pii"]    += stats["with_pii"]
        totals["annotations"] += stats["annotations"]

    print(f"\nSROIE done -> {out.resolve()}")
    print(f"  Total images      : {totals['total']}")
    print(f"  Images with PII   : {totals['with_pii']}")
    print(f"  Total annotations : {totals['annotations']}")
    for cls, cnt in sorted(totals["by_class"].items(), key=lambda x: -x[1]):
        print(f"    {cls}: {cnt}")
    print(f"\nNext: python data/download_cord.py")


if __name__ == "__main__":
    main()
