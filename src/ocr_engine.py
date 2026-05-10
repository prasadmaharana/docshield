import numpy as np

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        import logging
        logging.getLogger("ppocr").setLevel(logging.ERROR)
        from paddleocr import PaddleOCR
        _engine = PaddleOCR(use_angle_cls=True, lang="en")
    return _engine


def extract_text(image, bbox):
    """
    Crops bbox from image and runs PaddleOCR on the crop.
    image: PIL.Image (full page, RGB)
    bbox: [x1, y1, x2, y2]
    Returns extracted text string, or "" on any failure.
    """
    try:
        x1, y1, x2, y2 = bbox
        w, h = image.size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return ""

        crop_arr = np.array(image.crop((x1, y1, x2, y2)))
        engine = _get_engine()
        result = engine.ocr(crop_arr, cls=True)

        if not result or result[0] is None:
            return ""
        return " ".join(
            line[1][0] for line in result[0] if line and len(line) >= 2
        ).strip()
    except Exception:
        return ""
