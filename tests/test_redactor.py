"""Unit tests for src/redactor.py — no model loading required."""

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.redactor import redact_image, REDACTION_COLOR


def _blank(w=200, h=200, color=(255, 255, 255)):
    return Image.new("RGB", (w, h), color)


def _pixel(img, x, y):
    return img.getpixel((x, y))


class TestRedactImage:
    def test_returns_copy_not_original(self):
        img = _blank()
        out = redact_image(img, [])
        assert out is not img

    def test_no_detections_unchanged(self):
        img = _blank()
        out = redact_image(img, [])
        assert list(out.getbbox()) == list(img.getbbox()) and out.size == img.size and out.tobytes() == img.tobytes()

    def test_single_bbox_is_black(self):
        img = _blank()
        det = [{"class_name": "SSN", "confidence": 0.9, "bbox": [10, 10, 50, 30]}]
        out = redact_image(img, det)
        assert _pixel(out, 20, 20) == REDACTION_COLOR

    def test_outside_bbox_unchanged(self):
        img = _blank()
        det = [{"class_name": "SSN", "confidence": 0.9, "bbox": [10, 10, 50, 30]}]
        out = redact_image(img, det)
        assert _pixel(out, 100, 100) == (255, 255, 255)

    def test_multiple_bboxes(self):
        img = _blank()
        dets = [
            {"class_name": "SSN",   "confidence": 0.9, "bbox": [0,  0,  20, 20]},
            {"class_name": "EMAIL", "confidence": 0.8, "bbox": [50, 50, 80, 80]},
        ]
        out = redact_image(img, dets)
        assert _pixel(out, 10, 10)  == REDACTION_COLOR
        assert _pixel(out, 60, 60)  == REDACTION_COLOR
        assert _pixel(out, 35, 35)  == (255, 255, 255)

    def test_original_image_not_mutated(self):
        img = _blank()
        original_pixel = _pixel(img, 20, 20)
        det = [{"class_name": "SSN", "confidence": 0.9, "bbox": [10, 10, 50, 30]}]
        redact_image(img, det)
        assert _pixel(img, 20, 20) == original_pixel

    def test_rgb_mode_preserved(self):
        img = _blank()
        out = redact_image(img, [])
        assert out.mode == "RGB"
