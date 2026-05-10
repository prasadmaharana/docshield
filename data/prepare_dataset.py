"""
Merge synthetic + FUNSD + SROIE + CORD datasets, split 80/10/10, write YOLO config.

Scans each source dir for image+label pairs, shuffles, splits, copies into
data/dataset/{train,val,test}/{images,labels}/, then writes
train/configs/pii_detector.yaml for YOLOv8.

Usage:
    python data/prepare_dataset.py
    python data/prepare_dataset.py --synthetic data/synthetic --funsd data/funsd \\
        --sroie data/sroie --cord data/cord --out data/dataset
"""

import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml

CLASSES = [
    "PERSON_NAME", "SSN", "ACCOUNT_NUMBER", "ADDRESS",
    "PHONE", "EMAIL", "DATE_OF_BIRTH", "SIGNATURE",
]

SPLITS = {"train": 0.80, "val": 0.10, "test": 0.10}
SEED   = 42


# ── helpers ────────────────────────────────────────────────────────────────────

def collect_pairs(source_dir: Path) -> list[tuple[Path, Path]]:
    """
    Collect (image_path, label_path) pairs from a directory that has
    images/ and labels/ subdirs. Only keeps pairs where both files exist.
    """
    imgs_dir   = source_dir / "images"
    labels_dir = source_dir / "labels"
    if not imgs_dir.exists() or not labels_dir.exists():
        return []

    pairs = []
    for img in sorted(imgs_dir.glob("*.png")):
        lbl = labels_dir / (img.stem + ".txt")
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs


def split_pairs(pairs, ratios=SPLITS, seed=SEED):
    """Shuffle then split into train/val/test according to ratios."""
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios["train"])
    n_val   = int(n * ratios["val"])
    return {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train : n_train + n_val],
        "test":  shuffled[n_train + n_val :],
    }


def copy_split(split_name: str, pairs: list, out_dir: Path) -> Counter:
    """Copy image+label files into out_dir/{split}/images|labels/. Returns annotation counts."""
    imgs_dst   = out_dir / split_name / "images"
    labels_dst = out_dir / split_name / "labels"
    imgs_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    class_counts: Counter = Counter()
    for img_src, lbl_src in pairs:
        shutil.copy2(img_src, imgs_dst / img_src.name)
        shutil.copy2(lbl_src, labels_dst / lbl_src.name)
        for line in lbl_src.read_text().splitlines():
            parts = line.strip().split()
            if parts:
                class_counts[int(parts[0])] += 1

    return class_counts


def write_yaml(out_dir: Path, yaml_path: Path) -> None:
    """Write the pii_detector.yaml that YOLOv8 reads for training."""
    # YOLOv8 requires forward slashes even on Windows
    dataset_root = out_dir.resolve().as_posix()
    config = {
        "path":  dataset_root,
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc":    len(CLASSES),
        "names": CLASSES,
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def print_distribution(split_name: str, pairs: list, counts: Counter) -> None:
    total_anns = sum(counts.values())
    print(f"  {split_name:<6} {len(pairs):4d} images  {total_anns:5d} annotations")
    for cid, cname in enumerate(CLASSES):
        n = counts.get(cid, 0)
        bar = "#" * min(30, int(30 * n / max(total_anns, 1)))
        print(f"    {cid}  {cname:<16} {n:5d}  {bar}")


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DocShield — merge + split dataset")
    parser.add_argument("--synthetic", default="data/synthetic", help="Synthetic dataset dir")
    parser.add_argument("--funsd",     default="data/funsd",     help="FUNSD dataset dir")
    parser.add_argument("--sroie",     default="data/sroie",     help="SROIE dataset dir")
    parser.add_argument("--cord",      default="data/cord",      help="CORD dataset dir")
    parser.add_argument("--out",       default="data/dataset",   help="Output dataset dir")
    parser.add_argument("--yaml",      default="train/configs/pii_detector.yaml",
                        help="Path to write the YOLO dataset config")
    args = parser.parse_args()

    synthetic_dir = Path(args.synthetic)
    funsd_dir     = Path(args.funsd)
    sroie_dir     = Path(args.sroie)
    cord_dir      = Path(args.cord)
    out_dir       = Path(args.out)
    yaml_path     = Path(args.yaml)

    # ── clean output dir to remove stale files from previous runs ─────────────
    if out_dir.exists():
        import subprocess
        print(f"Cleaning existing dataset at {out_dir} ...")
        subprocess.run(
            f'rd /s /q "{out_dir.resolve()}"',
            shell=True, check=False,
        )
        if out_dir.exists():   # fallback if rd also failed
            shutil.rmtree(out_dir, ignore_errors=True)

    # ── collect ────────────────────────────────────────────────────────────────
    syn_pairs   = collect_pairs(synthetic_dir)
    funsd_pairs = collect_pairs(funsd_dir)
    sroie_pairs = collect_pairs(sroie_dir)
    cord_pairs  = collect_pairs(cord_dir)

    print(f"Sources found:")
    print(f"  Synthetic : {len(syn_pairs):4d} pairs  ({synthetic_dir})")
    print(f"  FUNSD     : {len(funsd_pairs):4d} pairs  ({funsd_dir})")
    print(f"  SROIE     : {len(sroie_pairs):4d} pairs  ({sroie_dir})")
    print(f"  CORD      : {len(cord_pairs):4d} pairs  ({cord_dir})")

    if len(syn_pairs) < 100:
        print(f"\n  WARNING: only {len(syn_pairs)} synthetic images found.")
        print("  Run `python data/generate_synthetic.py --count 500` for SSN/ACCOUNT/SIGNATURE coverage.")
        print("  Continuing with what's available...\n")

    all_pairs = syn_pairs + funsd_pairs + sroie_pairs + cord_pairs
    if not all_pairs:
        print("ERROR: no image/label pairs found. Check --synthetic and --funsd paths.")
        return

    print(f"\n  Total     : {len(all_pairs):4d} pairs\n")

    # ── split ──────────────────────────────────────────────────────────────────
    splits = split_pairs(all_pairs)

    # ── copy + report ──────────────────────────────────────────────────────────
    print(f"Copying into {out_dir}/ ...")
    all_counts: Counter = Counter()
    for split_name, pairs in splits.items():
        counts = copy_split(split_name, pairs, out_dir)
        print_distribution(split_name, pairs, counts)
        all_counts += counts
        print()

    # ── yaml ───────────────────────────────────────────────────────────────────
    write_yaml(out_dir, yaml_path)
    print(f"YOLO config written -> {yaml_path}")
    print(f"\nDataset ready. Total annotations across all splits:")
    for cid, cname in enumerate(CLASSES):
        print(f"  {cid}  {cname:<16} {all_counts.get(cid, 0):5d}")


if __name__ == "__main__":
    main()
