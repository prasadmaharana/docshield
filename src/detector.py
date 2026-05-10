from pathlib import Path

import torch
from ultralytics import YOLO

WEIGHTS_DEFAULT = Path(__file__).parent.parent / "models/best.pt"

CLASS_NAMES = [
    "PERSON_NAME",
    "SSN",
    "ACCOUNT_NUMBER",
    "ADDRESS",
    "PHONE",
    "EMAIL",
    "DATE_OF_BIRTH",
    "SIGNATURE",
]


class PIIDetector:
    def __init__(self, weights=None, device=None):
        path = Path(weights) if weights else WEIGHTS_DEFAULT
        if not path.exists():
            raise FileNotFoundError(
                f"Weights not found: {path}\nRun python train/train.py first."
            )
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = YOLO(str(path))

    def detect(self, image, conf=0.25):
        """
        image: PIL.Image (RGB)
        Returns list of {class_name, confidence, bbox: [x1, y1, x2, y2]}.
        """
        results = self._model(image, conf=conf, device=self._device, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    {
                        "class_name": CLASS_NAMES[int(box.cls[0])],
                        "confidence": round(float(box.conf[0]), 4),
                        "bbox": [round(x1), round(y1), round(x2), round(y2)],
                    }
                )
        return detections
