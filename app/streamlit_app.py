"""
DocShield — Streamlit UI
Upload a scanned PDF or image → detect PII → show side-by-side redaction → download.
"""

import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

import streamlit as st

# Make 'src' importable when launched from project root or app/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import Pipeline

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DocShield — PII Redactor",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("DocShield")
    st.caption("Offline PII redaction for scanned financial documents.")
    st.divider()

    conf = st.slider("Detection confidence threshold", 0.10, 0.90, 0.25, 0.05)
    run_ocr = st.checkbox("Extract OCR text (audit log)", value=True)

    st.divider()
    st.markdown("**8 PII classes detected**")
    for cls in [
        "PERSON_NAME", "SSN", "ACCOUNT_NUMBER", "ADDRESS",
        "PHONE", "EMAIL", "DATE_OF_BIRTH", "SIGNATURE",
    ]:
        st.markdown(f"- {cls}")


# ── Model loader (cached — loads once per server process) ────────────────────

@st.cache_resource(show_spinner="Loading PII detector model…")
def load_pipeline():
    return Pipeline()


# ── Processing ────────────────────────────────────────────────────────────────

def process(uploaded_file, conf, run_ocr):
    """
    Runs the pipeline on the uploaded file.
    Returns (original_pages, redacted_pages, audit_dict, output_bytes, out_name).
    """
    pipeline = load_pipeline()
    tmpdir = Path(tempfile.mkdtemp(prefix="docshield_"))
    try:
        suffix = Path(uploaded_file.name).suffix
        doc_path = tmpdir / f"input{suffix}"
        doc_path.write_bytes(uploaded_file.getvalue())

        original_pages = Pipeline.load_pages(doc_path)
        redacted_pages, audit = pipeline.redact(
            doc_path, output_dir=tmpdir, conf=conf, run_ocr=run_ocr
        )

        # Read output bytes before tmpdir is cleaned up
        out_files = sorted(tmpdir.glob("*_redacted_*"))
        if out_files:
            output_bytes = out_files[0].read_bytes()
            out_name = f"{Path(uploaded_file.name).stem}_redacted{out_files[0].suffix}"
        else:
            buf = io.BytesIO()
            redacted_pages[0].save(buf, format="PNG")
            output_bytes = buf.getvalue()
            out_name = f"{Path(uploaded_file.name).stem}_redacted.png"

        return original_pages, redacted_pages, audit, output_bytes, out_name
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Main UI ───────────────────────────────────────────────────────────────────

st.header("DocShield — Scanned Document PII Redactor")

uploaded = st.file_uploader(
    "Upload a scanned document",
    type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
)

if uploaded is not None:
    # Re-run processing only when the file or settings change
    file_key = f"{uploaded.name}_{uploaded.size}_{conf}_{run_ocr}"
    if st.session_state.get("file_key") != file_key:
        with st.spinner("Running PII detection…"):
            try:
                result = process(uploaded, conf, run_ocr)
                st.session_state.result = result
                st.session_state.file_key = file_key
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.stop()

    original_pages, redacted_pages, audit, output_bytes, out_name = (
        st.session_state.result
    )

    # ── Summary bar ──────────────────────────────────────────────────────────
    pii_count = audit["pii_count"]
    n_pages = len(redacted_pages)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("PII regions found", pii_count)
    col_b.metric("Pages processed", n_pages)
    col_c.metric("Document SHA-256", audit["document_sha256"][:12] + "…")

    st.divider()

    # ── Side-by-side viewer ───────────────────────────────────────────────────
    if n_pages > 1:
        page_no = st.selectbox("Page", list(range(1, n_pages + 1))) - 1
    else:
        page_no = 0

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(original_pages[page_no], use_container_width=True)
    with col2:
        st.subheader("Redacted")
        st.image(redacted_pages[page_no], use_container_width=True)

    st.divider()

    # ── Detection table ───────────────────────────────────────────────────────
    if pii_count > 0:
        st.subheader(f"Detections ({pii_count})")
        page_dets = [
            d for d in audit["detections"] if d["page"] == page_no + 1
        ]
        if page_dets:
            import pandas as pd
            df = pd.DataFrame(
                [
                    {
                        "Class": d["class"],
                        "Confidence": f"{d['confidence']:.1%}",
                        "BBox [x1,y1,x2,y2]": str(d["bbox"]),
                        "OCR text": d.get("ocr_text", ""),
                    }
                    for d in page_dets
                ]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No PII detected on this page.")
    else:
        st.success("No PII detected in this document.")

    # ── Audit log ─────────────────────────────────────────────────────────────
    with st.expander("Audit log (JSON)"):
        st.json(audit)

    # ── Downloads ─────────────────────────────────────────────────────────────
    st.divider()
    dl_col1, dl_col2 = st.columns(2)

    mime = (
        "application/pdf"
        if out_name.endswith(".pdf")
        else "image/png"
    )
    dl_col1.download_button(
        label="Download redacted document",
        data=output_bytes,
        file_name=out_name,
        mime=mime,
    )

    audit_bytes = json.dumps(audit, indent=2, ensure_ascii=False).encode("utf-8")
    dl_col2.download_button(
        label="Download audit log (JSON)",
        data=audit_bytes,
        file_name=f"{Path(uploaded.name).stem}_audit.json",
        mime="application/json",
    )

else:
    st.info("Upload a scanned PDF or image to begin.")
