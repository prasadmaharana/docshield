from PIL import ImageDraw

REDACTION_COLOR = (0, 0, 0)


def redact_image(image, detections):
    """Returns a copy of image with solid black rectangles over all detection bboxes."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        draw.rectangle([x1, y1, x2, y2], fill=REDACTION_COLOR)
    return out
