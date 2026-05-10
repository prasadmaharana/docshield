"""
Download and convert FUNSD to YOLO PII annotation format for DocShield.

FUNSD (Form Understanding in Noisy Scanned Documents):
  149 train + 50 test real scanned government / administrative forms.
  HuggingFace: guillaumejaume/FUNSD  (public, no auth required)

Annotation strategy:
  Token-level NER tags are merged into entity spans. ANSWER spans are
  classified into DocShield PII classes using regex matching and
  nearest-preceding-QUESTION keyword heuristics. Unclassifiable spans
  are silently dropped; images with zero PII hits are still saved (they
  serve as background-negative examples during training).

Output mirrors data/synthetic/ so prepare_dataset.py can merge both:
    <output>/
        images/  *.png
        labels/  *.txt   (YOLO format; empty = no PII in this image)

Usage:
    python data/download_funsd.py
    python data/download_funsd.py --output data/funsd
"""

import argparse
import re
import sys
from pathlib import Path

# ── PII class map (must match generate_synthetic.py) ──────────────────────────

CLASSES = [
    "PERSON_NAME", "SSN", "ACCOUNT_NUMBER", "ADDRESS",
    "PHONE", "EMAIL", "DATE_OF_BIRTH", "SIGNATURE",
]
CID = {c: i for i, c in enumerate(CLASSES)}


# ── text-based PII classifier ─────────────────────────────────────────────────

_SSN_RE   = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
_EIN_RE   = re.compile(r'\b\d{2}-\d{7}\b')
_EMAIL_RE = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.]+\b')
_PHONE_RE = re.compile(r'\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
_DATE_RE  = re.compile(r'\b(0?[1-9]|1[012])[/\-](0?[1-9]|[12]\d|3[01])[/\-](19|20)?\d{2}\b')
_ACCT_RE  = re.compile(r'\b\d{7,20}\b')
_NAME_RE  = re.compile(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z.]+){1,3}$')
_ADDR_RE  = re.compile(
    r'\b\d{1,5}\s+[A-Za-z]+\s+(?:St|Ave|Rd|Blvd|Dr|Ln|Way|Ct|Pkwy|Hwy)\b', re.I
)

# Question keywords that strongly imply a PII class for the linked answer
_Q_MAP: dict[str, list[str]] = {
    "PERSON_NAME":    ["name", "employee", "employer", "recipient",
                       "taxpayer", "beneficiary", "payee", "payer", "claimant"],
    "SSN":            ["ssn", "social security", "ss #", "s.s.n"],
    "ACCOUNT_NUMBER": ["account", "acct", "ein", "tin", "routing",
                       "id number", "employer id", "federal id", "policy"],
    "ADDRESS":        ["address", "street", "city", "state", "zip",
                       "location", "residence"],
    "PHONE":          ["phone", "telephone", "tel", "fax", "mobile", "cell"],
    "EMAIL":          ["email", "e-mail"],
    "DATE_OF_BIRTH":  ["birth", "dob", "born", "date of birth", "d.o.b"],
}


def classify_pii(text: str, q_context: str = "") -> str | None:
    """
    Classify a text string into one of the 8 PII classes.
    Returns the class name string or None if no match.
    Priority: strict regex patterns > question-keyword > loose text patterns.
    """
    t = text.strip()
    q = q_context.lower()

    if _SSN_RE.search(t):                          return "SSN"
    if _EIN_RE.search(t):                          return "ACCOUNT_NUMBER"
    if _EMAIL_RE.search(t):                        return "EMAIL"
    if _PHONE_RE.search(t) and len(t) < 25:        return "PHONE"
    if _DATE_RE.search(t):                         return "DATE_OF_BIRTH"

    for cls, keywords in _Q_MAP.items():
        if any(kw in q for kw in keywords):
            if cls == "PERSON_NAME":
                # Require at least 2 words and reasonable length for a name
                if not (1 < len(t.split()) <= 5 and len(t) < 50):
                    continue
            return cls

    # Loose fallbacks (no question context)
    if _ACCT_RE.search(t) and len(t) < 22:        return "ACCOUNT_NUMBER"
    if _NAME_RE.match(t) and len(t) < 45:         return "PERSON_NAME"
    if _ADDR_RE.search(t):                        return "ADDRESS"

    return None


# ── span extraction ────────────────────────────────────────────────────────────

def extract_qa_spans(words, bboxes, ner_tags, B_ANS, I_ANS, B_QUE, I_QUE):
    """
    Merge consecutive token-level NER tags into entity spans.
    For each ANSWER span, captures the text of the nearest preceding
    QUESTION span as question context for the classifier.

    Returns list of (span_text, list_of_token_bboxes, question_context_str).
    """
    # First pass: collect all spans in document order
    spans = []   # (start_idx, end_idx, label_str)
    i = 0
    while i < len(ner_tags):
        tag = ner_tags[i]
        if tag == B_ANS:
            j = i + 1
            while j < len(ner_tags) and ner_tags[j] == I_ANS:
                j += 1
            spans.append((i, j, "ANSWER"))
            i = j
        elif tag == B_QUE:
            j = i + 1
            while j < len(ner_tags) and ner_tags[j] == I_QUE:
                j += 1
            spans.append((i, j, "QUESTION"))
            i = j
        else:
            i += 1

    # Second pass: pair each ANSWER with its nearest preceding QUESTION
    results = []
    for idx, (start, end, label) in enumerate(spans):
        if label != "ANSWER":
            continue

        q_text = ""
        for ps, pe, pl in reversed(spans[:idx]):
            if pl == "QUESTION":
                q_text = " ".join(words[ps:pe])
                break

        results.append(
            (" ".join(words[start:end]), bboxes[start:end], q_text)
        )

    return results


def merge_bbox(boxes: list) -> list:
    """Minimum enclosing axis-aligned bbox from a list of [x1,y1,x2,y2] boxes."""
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


# ── coordinate normalisation ───────────────────────────────────────────────────

def make_converter(bboxes, img_w: int, img_h: int):
    """
    Auto-detect the bbox coordinate space and return a conversion function.

    FUNSD via datasets library often ships bboxes normalised to [0, 1000]
    (LayoutLM convention). The original JSONs use pixel coords. Both cases
    are handled by inspecting the maximum coordinate value.

    Returns: callable([x1,y1,x2,y2]) -> (cx, cy, w, h) in [0,1]
    """
    if not bboxes:
        return lambda b: (0.5, 0.5, 0.05, 0.05)

    flat = [v for b in bboxes for v in b]
    max_val = max(flat)

    if max_val <= 1.0:
        sx, sy = float(img_w), float(img_h)      # [0,1] space
    elif max_val <= 1001:
        sx, sy = img_w / 1000.0, img_h / 1000.0  # [0,1000] space (LayoutLM)
    else:
        sx, sy = 1.0, 1.0                         # pixel space

    def convert(box):
        x1, y1, x2, y2 = box
        px1, py1 = x1 * sx, y1 * sy
        px2, py2 = x2 * sx, y2 * sy
        cx = (px1 + px2) / 2 / img_w
        cy = (py1 + py2) / 2 / img_h
        w  = (px2 - px1) / img_w
        h  = (py2 - py1) / img_h
        return (
            max(0.0, min(1.0, cx)),
            max(0.0, min(1.0, cy)),
            max(0.001, min(1.0, w)),
            max(0.001, min(1.0, h)),
        )

    return convert


# ── split converter ────────────────────────────────────────────────────────────

def convert_split(split_data, features, out_dir: Path, prefix: str) -> dict:
    """Process one dataset split, write images + YOLO labels. Returns stats."""
    imgs_dir   = out_dir / "images"
    labels_dir = out_dir / "labels"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Resolve tag indices from the dataset schema (robust to reordering)
    tag_names = features["ner_tags"].feature.names
    t2i = {t: i for i, t in enumerate(tag_names)}
    B_ANS = t2i["B-ANSWER"]
    I_ANS = t2i["I-ANSWER"]
    B_QUE = t2i["B-QUESTION"]
    I_QUE = t2i["I-QUESTION"]

    stats = {"total": 0, "with_pii": 0, "annotations": 0}

    for idx, example in enumerate(split_data):
        doc_id  = f"{prefix}_{idx:04d}"
        image   = example["image"]
        img_w, img_h = image.size

        to_yolo = make_converter(example["bboxes"], img_w, img_h)

        spans = extract_qa_spans(
            example["words"], example["bboxes"], example["ner_tags"],
            B_ANS, I_ANS, B_QUE, I_QUE,
        )

        yolo_lines = []
        for span_text, span_boxes, q_text in spans:
            cls_name = classify_pii(span_text, q_text)
            if cls_name is None:
                continue
            merged = merge_bbox(span_boxes)
            cx, cy, w, h = to_yolo(merged)
            yolo_lines.append(f"{CID[cls_name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        image.convert("RGB").save(imgs_dir / f"{doc_id}.png")

        with open(labels_dir / f"{doc_id}.txt", "w") as f:
            if yolo_lines:
                f.write("\n".join(yolo_lines) + "\n")

        stats["total"]       += 1
        stats["with_pii"]    += bool(yolo_lines)
        stats["annotations"] += len(yolo_lines)

        if (idx + 1) % 25 == 0 or idx == 0:
            print(
                f"  [{idx+1:3d}]  {doc_id}  {img_w}x{img_h}  "
                f"pii_hits={len(yolo_lines)}",
                flush=True,
            )

    return stats


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DocShield — download FUNSD and convert to YOLO PII format"
    )
    parser.add_argument(
        "--output", default="data/funsd",
        help="Output directory (default: data/funsd)",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package is not installed.")
        print("       Run:  pip install datasets")
        sys.exit(1)

    out = Path(args.output)
    # guillaumejaume/FUNSD was removed from the Hub; use the maintained mirror
    REPO = "nielsr/funsd"
    print(f"Downloading FUNSD from HuggingFace ({REPO})...")
    ds = load_dataset(REPO)

    print(f"Train: {len(ds['train'])} samples   Test: {len(ds['test'])} samples\n")

    totals = {"total": 0, "with_pii": 0, "annotations": 0}
    for split_name, prefix in [("train", "tr"), ("test", "te")]:
        print(f"-- {split_name.upper()} --")
        stats = convert_split(
            ds[split_name], ds[split_name].features, out, prefix
        )
        print(
            f"  {stats['total']} images | "
            f"{stats['with_pii']} with PII | "
            f"{stats['annotations']} annotations\n"
        )
        for k in totals:
            totals[k] += stats[k]

    print(f"FUNSD conversion complete -> {out}/")
    print(f"  Total images       : {totals['total']}")
    print(f"  Images with PII    : {totals['with_pii']}")
    print(f"  Total annotations  : {totals['annotations']}")
    print(f"  Class map: {', '.join(f'{i}={c}' for i, c in enumerate(CLASSES))}")


if __name__ == "__main__":
    main()
