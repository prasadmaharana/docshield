# DocShield

Computer vision pipeline that detects and redacts PII from scanned financial document images — fully offline, with an auditable JSON trail per document.

**Differentiator from text-based tools (e.g. Microsoft Presidio):** works directly on raw image input (scanned PDFs, photos, handwritten forms), not just extracted text.

## Status

> **Work in progress.** The pipeline is end-to-end functional and the model achieves strong metrics on the test set, but real-world performance on unseen bank statement layouts is inconsistent. See [Known limitations](#known-limitations).

## What works

- **End-to-end pipeline** — PDF/image in, redacted output + JSON audit log out
- **Streamlit app** — upload, side-by-side viewer, detection table, download
- **8 PII classes** — SSN, EMAIL, SIGNATURE, ACCOUNT_NUMBER detected with high reliability (mAP50 ≥ 0.99)
- **Synthetic document coverage** — bank statements, KYC forms, cheques, contracts, loan applications generated with realistic scan simulation
- **Audit trail** — per-document SHA-256, timestamps, per-detection class/confidence/OCR text
- **ONNX export** — CPU-compatible model for deployment without GPU

## Known limitations

**Real bank statement detection quality is inconsistent.** The model was trained on synthetic bank statements and real scanned receipts (SROIE + CORD datasets). Real bank statements from production sources have different visual layouts — the model tends to:
- Miss PII in the account-holder header block (name, address, account number at the top)
- Produce false positives on transaction description rows that contain number sequences

**Root cause:** There are no large, publicly available labeled datasets of real bank statements. The synthetic generator produces realistic-looking documents but the visual gap to real-world statements is non-trivial. The fix is more annotated real examples — even 50–100 manually labeled bank statement images would significantly improve this.

**DATE_OF_BIRTH is the weakest class** (mAP50 0.91). SROIE/CORD receipts label transaction dates as dates, which teaches the model a different visual pattern than an actual date-of-birth field on a KYC form.

**OCR requires a separate install.** PaddleOCR needs `paddlepaddle` as a backend, which is not bundled in `requirements.txt` by default because the GPU/CPU variant depends on your hardware. OCR is used only for the audit log — redaction works without it.

## PII classes detected

| Class | Test mAP50 | Notes |
|-------|-----------|-------|
| SSN | 1.00 | Pattern-distinct, very reliable |
| EMAIL | 1.00 | Pattern-distinct, very reliable |
| SIGNATURE | 1.00 | Visual blob, very reliable |
| ACCOUNT_NUMBER | 1.00 | Strong |
| PHONE | 0.95 | Strong |
| ADDRESS | 0.95 | Strong on trained layouts |
| PERSON_NAME | 0.97 | Varies by document type |
| DATE_OF_BIRTH | 0.91 | Weakest — see limitations above |

## Supported document types

| Type | Status |
|------|--------|
| Synthetic bank statements | Works well |
| KYC / identity forms | Works well |
| Cheques | Works well |
| Contracts | Works well |
| Scanned receipts (SROIE/CORD style) | Works well |
| Real bank statements (production) | Inconsistent — see limitations |
| Handwritten forms | Untested |

## Model

YOLOv8n fine-tuned on 7,022 images:
- 5,000 synthetic financial documents (bank statements, KYC, cheques, contracts, loan applications)
- 973 real scanned receipts — SROIE 2019
- 850 real scanned receipts — CORD v2
- 199 real scanned forms — FUNSD

| Metric | Value |
|--------|-------|
| mAP50 (test) | 0.971 |
| mAP50-95 | 0.943 |
| Precision | 0.981 |
| Recall | 0.967 |

*Test set is mixed real + synthetic, same distribution as training. Performance on out-of-distribution real documents will be lower.*

## Installation

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install PyTorch (adjust index URL for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Install DocShield dependencies
pip install -r requirements.txt

# 4. (Optional) OCR support for audit log text extraction
pip install paddlepaddle   # CPU
# pip install paddlepaddle-gpu  # GPU
```

## Quick start

### Python API

```python
from src.pipeline import Pipeline

pipeline = Pipeline()  # loads models/best.pt
redacted_pages, audit = pipeline.redact(
    "path/to/document.pdf",
    output_dir="output/",
    conf=0.25,
    run_ocr=False,   # set True if paddlepaddle is installed
)
# redacted_pages: list of PIL Images
# audit: dict with timestamp, SHA256, per-detection records
# Files written: output/*_redacted_*.png + output/*_audit_*.json
```

### Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Upload a scanned PDF or image → redaction boxes appear → download the redacted document and audit log.

## Project structure

```
docshield/
├── src/                    # Core library
│   ├── pipeline.py         # End-to-end orchestration
│   ├── detector.py         # YOLOv8 inference wrapper
│   ├── redactor.py         # Pillow redaction
│   ├── ocr_engine.py       # PaddleOCR (audit log only)
│   └── audit_logger.py     # JSON audit trail
├── app/
│   └── streamlit_app.py    # Web UI
├── models/
│   ├── best.pt             # YOLOv8n weights (~6 MB)
│   └── pii_detector.onnx   # ONNX export for CPU inference (~12 MB)
├── train/                  # Training scripts
├── data/                   # Dataset download + generation scripts
└── tests/                  # Unit tests (19 passing)
```

## Training

```bash
# Download real datasets
python data/download_funsd.py
python data/download_sroie.py
python data/download_cord.py

# Generate synthetic documents
python data/generate_synthetic.py --count 5000 --output data/synthetic

# Merge + split (80/10/10)
python data/prepare_dataset.py

# Train — requires CUDA GPU (tested on GTX 1660, ~2 hrs)
python train/train.py

# Validate on held-out test split
python train/validate.py

# Export to ONNX
python train/export_onnx.py
```

## Running tests

```bash
pip install pytest
pytest tests/
```

## What's next

- [ ] Manually annotate 50–100 real bank statement images to close the domain gap
- [ ] Retrain with annotated real bank statements mixed in
- [ ] Add confidence calibration / post-processing NMS tuning per document type
- [ ] HuggingFace Spaces deployment using ONNX model for CPU inference

## License

MIT — see [LICENSE](LICENSE).
