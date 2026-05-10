"""
Integration-lite tests for src/pipeline.py.

The YOLOv8 detector is mocked so tests run without GPU or model weights.
OCR is also mocked to keep tests fast and offline.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.pipeline import Pipeline

# ── helpers ────────────────────────────────────────────────────────────────────

FAKE_DETECTIONS = [
    {"class_name": "PERSON_NAME",   "confidence": 0.95, "bbox": [10, 10, 100, 30]},
    {"class_name": "ACCOUNT_NUMBER","confidence": 0.88, "bbox": [10, 50, 200, 70]},
]


def _make_pipeline():
    """Return a Pipeline with a mocked detector (no model loading)."""
    with patch("src.pipeline.PIIDetector") as MockDetector:
        instance = MockDetector.return_value
        instance.detect.return_value = FAKE_DETECTIONS
        p = Pipeline.__new__(Pipeline)
        p._detector = instance
        return p, instance


def _small_image(w=300, h=400):
    return Image.new("RGB", (w, h), color=(240, 240, 240))


# ── load_pages ─────────────────────────────────────────────────────────────────

class TestLoadPages:
    def test_png_returns_single_page(self, tmp_path):
        img = _small_image()
        p = tmp_path / "doc.png"
        img.save(p)
        pages = Pipeline.load_pages(p)
        assert len(pages) == 1
        assert pages[0].mode == "RGB"

    def test_jpg_returns_single_page(self, tmp_path):
        img = _small_image()
        p = tmp_path / "doc.jpg"
        img.save(p)
        pages = Pipeline.load_pages(p)
        assert len(pages) == 1


# ── redact ─────────────────────────────────────────────────────────────────────

class TestRedact:
    def test_output_file_created(self, tmp_path):
        pipeline, _ = _make_pipeline()
        img = _small_image()
        src = tmp_path / "input.png"
        img.save(src)

        _, audit = pipeline.redact(src, output_dir=tmp_path, run_ocr=False)

        out_files = list(tmp_path.glob("*_redacted_*.png"))
        assert len(out_files) == 1

    def test_audit_log_created(self, tmp_path):
        pipeline, _ = _make_pipeline()
        img = _small_image()
        src = tmp_path / "input.png"
        img.save(src)

        _, audit = pipeline.redact(src, output_dir=tmp_path, run_ocr=False)

        log_files = list(tmp_path.glob("*_audit_*.json"))
        assert len(log_files) == 1
        data = json.loads(log_files[0].read_text())
        assert "document_sha256" in data
        assert "pii_count" in data

    def test_pii_count_matches_detections(self, tmp_path):
        pipeline, _ = _make_pipeline()
        img = _small_image()
        src = tmp_path / "input.png"
        img.save(src)

        _, audit = pipeline.redact(src, output_dir=tmp_path, run_ocr=False)

        assert audit["pii_count"] == len(FAKE_DETECTIONS)

    def test_detection_classes_in_audit(self, tmp_path):
        pipeline, _ = _make_pipeline()
        img = _small_image()
        src = tmp_path / "input.png"
        img.save(src)

        _, audit = pipeline.redact(src, output_dir=tmp_path, run_ocr=False)

        classes = {d["class"] for d in audit["detections"]}
        assert "PERSON_NAME" in classes
        assert "ACCOUNT_NUMBER" in classes

    def test_redacted_pages_returned(self, tmp_path):
        pipeline, _ = _make_pipeline()
        img = _small_image()
        src = tmp_path / "input.png"
        img.save(src)

        pages, _ = pipeline.redact(src, output_dir=tmp_path, run_ocr=False)

        assert len(pages) == 1
        assert isinstance(pages[0], Image.Image)

    def test_redacted_pixels_are_black(self, tmp_path):
        pipeline, _ = _make_pipeline()
        img = _small_image()
        src = tmp_path / "input.png"
        img.save(src)

        pages, _ = pipeline.redact(src, output_dir=tmp_path, run_ocr=False)

        x1, y1, x2, y2 = FAKE_DETECTIONS[0]["bbox"]
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        assert pages[0].getpixel((mid_x, mid_y)) == (0, 0, 0)

    def test_ocr_called_when_enabled(self, tmp_path):
        pipeline, detector = _make_pipeline()
        img = _small_image()
        src = tmp_path / "input.png"
        img.save(src)

        with patch("src.pipeline.extract_text", return_value="mocked text") as mock_ocr:
            pipeline.redact(src, output_dir=tmp_path, run_ocr=True)
            assert mock_ocr.call_count == len(FAKE_DETECTIONS)

    def test_ocr_skipped_when_disabled(self, tmp_path):
        pipeline, _ = _make_pipeline()
        img = _small_image()
        src = tmp_path / "input.png"
        img.save(src)

        with patch("src.pipeline.extract_text") as mock_ocr:
            pipeline.redact(src, output_dir=tmp_path, run_ocr=False)
            mock_ocr.assert_not_called()

    def test_output_dir_created_if_missing(self, tmp_path):
        pipeline, _ = _make_pipeline()
        img = _small_image()
        src = tmp_path / "input.png"
        img.save(src)
        out = tmp_path / "nested" / "output"

        pipeline.redact(src, output_dir=out, run_ocr=False)

        assert out.exists()

    def test_page_numbers_in_audit(self, tmp_path):
        pipeline, _ = _make_pipeline()
        img = _small_image()
        src = tmp_path / "input.png"
        img.save(src)

        _, audit = pipeline.redact(src, output_dir=tmp_path, run_ocr=False)

        for det in audit["detections"]:
            assert det["page"] == 1
