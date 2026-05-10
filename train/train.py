"""
Fine-tune YOLOv8n on the merged PII dataset for DocShield.

Downloads pretrained yolov8n.pt weights (~6MB) on first run.
Saves best checkpoint to runs/detect/pii_detector/weights/best.pt

Usage:
    python train/train.py
    python train/train.py --epochs 50 --batch 8
    python train/train.py --resume  # resume from last checkpoint
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="DocShield — train YOLOv8n PII detector")
    parser.add_argument("--data",    default="train/configs/pii_detector.yaml",
                        help="Path to dataset YAML (written by prepare_dataset.py)")
    parser.add_argument("--epochs",  type=int,   default=50)
    parser.add_argument("--batch",   type=int,   default=16,
                        help="Batch size — 16 uses ~3.5GB VRAM on GTX 1660")
    parser.add_argument("--imgsz",   type=int,   default=640)
    parser.add_argument("--device",  default="0",
                        help="CUDA device index or 'cpu'")
    parser.add_argument("--resume",  action="store_true",
                        help="Resume training from runs/detect/pii_detector/weights/last.pt")
    parser.add_argument("--name",    default="pii_detector",
                        help="Run name under runs/detect/")
    args = parser.parse_args()

    # Validate config exists
    yaml_path = Path(args.data)
    if not yaml_path.exists():
        print(f"ERROR: dataset config not found at {yaml_path}")
        print("       Run `python data/prepare_dataset.py` first.")
        return

    from ultralytics import YOLO

    if args.resume:
        last_ckpt = Path(f"runs/detect/{args.name}/weights/last.pt")
        if not last_ckpt.exists():
            print(f"ERROR: no checkpoint to resume from at {last_ckpt}")
            return
        print(f"Resuming from {last_ckpt}")
        model = YOLO(str(last_ckpt))
    else:
        # yolov8n.pt auto-downloads from Ultralytics on first run (~6MB)
        model = YOLO("yolov8n.pt")

    print(f"\nStarting training:")
    print(f"  Config  : {yaml_path.resolve()}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch}")
    print(f"  Img size: {args.imgsz}")
    print(f"  Device  : {args.device}")
    print(f"  Run name: {args.name}\n")

    results = model.train(
        data=str(yaml_path.resolve()),  # absolute path avoids CWD ambiguity
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,           # final lr = lr0 * lrf
        warmup_epochs=3,
        patience=10,         # early stop if no improvement for 10 epochs
        augment=True,
        mosaic=1.0,         # mosaic augmentation — helps with varied layouts
        mixup=0.1,
        copy_paste=0.0,
        degrees=2.0,        # matches scan simulation rotation range
        translate=0.05,
        scale=0.3,
        fliplr=0.0,         # documents aren't mirrored
        flipud=0.0,
        project=str(Path("runs").resolve()),
        name=args.name,
        exist_ok=True,
        save=True,
        save_period=10,     # checkpoint every 10 epochs
        val=True,
        plots=True,
        verbose=True,
    )

    best = Path("runs") / args.name / "weights" / "best.pt"
    print(f"\nTraining complete.")
    print(f"  Best checkpoint : {best.resolve()}")
    print(f"  mAP50           : {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"  mAP50-95        : {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    print(f"\nNext step: python train/validate.py")


if __name__ == "__main__":
    main()
