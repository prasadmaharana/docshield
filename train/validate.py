"""
Validate the trained PII detector and print per-class metrics.

Usage:
    python train/validate.py
    python train/validate.py --weights runs/pii_detector/weights/best.pt
    python train/validate.py --split test
"""

import argparse
from pathlib import Path


CLASSES = [
    "PERSON_NAME", "SSN", "ACCOUNT_NUMBER", "ADDRESS",
    "PHONE", "EMAIL", "DATE_OF_BIRTH", "SIGNATURE",
]

PASS_THRESHOLD = 0.70   # mAP50 target from CLAUDE.md


def main():
    parser = argparse.ArgumentParser(description="DocShield — validate PII detector")
    parser.add_argument(
        "--weights", default="models/best.pt",
        help="Path to trained .pt weights",
    )
    parser.add_argument(
        "--data", default="train/configs/pii_detector.yaml",
        help="Dataset YAML used during training",
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"],
        help="Which split to evaluate on",
    )
    parser.add_argument("--imgsz",  type=int,   default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold for predictions",
    )
    parser.add_argument(
        "--iou", type=float, default=0.50,
        help="IoU threshold for NMS",
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    yaml_path    = Path(args.data)

    if not weights_path.exists():
        print(f"ERROR: weights not found at {weights_path}")
        print("       Run `python train/train.py` first.")
        return

    if not yaml_path.exists():
        print(f"ERROR: dataset config not found at {yaml_path}")
        return

    from ultralytics import YOLO

    print(f"Loading weights: {weights_path.resolve()}")
    model = YOLO(str(weights_path))

    print(f"Evaluating on '{args.split}' split ...\n")
    metrics = model.val(
        data=str(yaml_path.resolve()),
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        verbose=False,
    )

    # ── overall metrics ────────────────────────────────────────────────────────
    map50    = metrics.box.map50
    map5095  = metrics.box.map
    p_mean   = metrics.box.mp
    r_mean   = metrics.box.mr

    print("=" * 62)
    print(f"  Overall results  (split={args.split})")
    print("=" * 62)
    print(f"  mAP50       : {map50:.4f}  {'PASS' if map50 >= PASS_THRESHOLD else 'BELOW TARGET (need >= 0.70)'}")
    print(f"  mAP50-95    : {map5095:.4f}")
    print(f"  Precision   : {p_mean:.4f}")
    print(f"  Recall      : {r_mean:.4f}")
    print()

    # ── per-class breakdown ────────────────────────────────────────────────────
    ap50_per_class = metrics.box.ap50        # shape: (nc,)
    p_per_class    = metrics.box.p           # precision per class
    r_per_class    = metrics.box.r           # recall per class

    print(f"  {'Class':<18} {'P':>6} {'R':>6} {'mAP50':>8}  Bar")
    print("  " + "-" * 58)
    for i, cls in enumerate(CLASSES):
        ap   = float(ap50_per_class[i]) if i < len(ap50_per_class) else 0.0
        p    = float(p_per_class[i])    if i < len(p_per_class)    else 0.0
        r    = float(r_per_class[i])    if i < len(r_per_class)    else 0.0
        bar  = "#" * int(ap * 30)
        flag = " <-- weak" if ap < 0.50 else ""
        print(f"  {cls:<18} {p:>6.3f} {r:>6.3f} {ap:>8.3f}  {bar}{flag}")

    print()

    # ── recommendations ───────────────────────────────────────────────────────
    weak = [(CLASSES[i], float(ap50_per_class[i]))
            for i in range(len(CLASSES)) if float(ap50_per_class[i]) < 0.50]
    if weak:
        print("  Classes below 0.50 mAP50 — consider:")
        for cls, ap in sorted(weak, key=lambda x: x[1]):
            print(f"    {cls:<18} {ap:.3f} — add more synthetic examples of this class")
    else:
        print("  All classes >= 0.50 mAP50.")

    if map50 >= PASS_THRESHOLD:
        print(f"\n  Target mAP50 >= {PASS_THRESHOLD} ACHIEVED. Ready for Phase 5 (ONNX export).")
        print("  Next: python train/export_onnx.py")
    else:
        print(f"\n  Target mAP50 >= {PASS_THRESHOLD} not yet reached ({map50:.4f}).")
        print("  Suggestion: ensure 5000 synthetic images are generated and retrain.")


if __name__ == "__main__":
    main()
