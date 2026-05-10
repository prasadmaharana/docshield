"""
Export the trained YOLOv8n PII detector to ONNX format for CPU cloud inference.

The exported model runs on CPU (no CUDA required) — suitable for HuggingFace Spaces
and any cloud container without a GPU.

Usage:
    python train/export_onnx.py
    python train/export_onnx.py --weights runs/detect/runs/pii_detector/weights/best.pt
    python train/export_onnx.py --dynamic   # dynamic batch axis (recommended for serving)
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="DocShield — export best.pt to ONNX")
    parser.add_argument(
        "--weights",
        default="models/best.pt",
        help="Path to trained .pt checkpoint",
    )
    parser.add_argument(
        "--output", default="models/pii_detector.onnx",
        help="Destination path for the exported ONNX file",
    )
    parser.add_argument("--imgsz",   type=int,  default=640)
    parser.add_argument("--dynamic", action="store_true",
                        help="Export with dynamic batch dimension")
    parser.add_argument("--simplify", action="store_true", default=True,
                        help="Run onnxsim to simplify the graph (default: True)")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    output_path  = Path(args.output)

    if not weights_path.exists():
        print(f"ERROR: weights not found at {weights_path}")
        print("       Run `python train/train.py` first.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    print(f"Loading: {weights_path.resolve()}")
    model = YOLO(str(weights_path))

    print(f"Exporting to ONNX (imgsz={args.imgsz}, dynamic={args.dynamic}) ...")
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        dynamic=args.dynamic,
        simplify=args.simplify,
        opset=17,       # ONNX opset 17 — broad runtime support
        half=False,     # FP32 for CPU compatibility
    )

    # Ultralytics saves next to the .pt file; copy to models/ if different
    exported_path = Path(exported)
    if exported_path.resolve() != output_path.resolve():
        import shutil
        shutil.copy2(exported_path, output_path)
        print(f"Copied to {output_path.resolve()}")

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\nExport complete.")
    print(f"  ONNX model : {output_path.resolve()}")
    print(f"  Size       : {size_mb:.1f} MB")
    print(f"\nNext step: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
