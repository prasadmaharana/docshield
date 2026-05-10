import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_audit_log(doc_path, detections, output_path, log_path):
    """
    Writes a JSON audit record for a processed document.
    detections: list of {class_name, confidence, bbox, page, ocr_text (optional)}
    Returns the record dict.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "document": str(Path(doc_path).resolve()),
        "document_sha256": _sha256(doc_path),
        "redacted_output": str(Path(output_path).resolve()),
        "pii_count": len(detections),
        "detections": [
            {
                "page": d.get("page", 1),
                "class": d["class_name"],
                "confidence": d["confidence"],
                "bbox": d["bbox"],
                "ocr_text": d.get("ocr_text", ""),
            }
            for d in detections
        ],
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    return record
