from datetime import datetime
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

from .audit_logger import write_audit_log
from .detector import PIIDetector
from .ocr_engine import extract_text
from .redactor import redact_image


class Pipeline:
    def __init__(self, weights=None, device=None):
        self._detector = PIIDetector(weights=weights, device=device)

    @staticmethod
    def load_pages(doc_path):
        """Load a PDF or image as a list of RGB PIL Images."""
        suffix = Path(doc_path).suffix.lower()
        if suffix == ".pdf":
            pdf = pdfium.PdfDocument(str(doc_path))
            return [pdf[i].render(scale=2.0).to_pil() for i in range(len(pdf))]
        return [Image.open(doc_path).convert("RGB")]

    def redact(self, doc_path, output_dir=None, conf=0.25, run_ocr=True):
        """
        Full pipeline: load → detect → (OCR) → redact → save output + audit log.

        Returns:
            redacted_pages: list[PIL.Image]
            audit:          dict  (same data written to audit JSON)
        """
        doc_path = Path(doc_path)
        output_dir = Path(output_dir) if output_dir else doc_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = doc_path.stem
        pages = self.load_pages(doc_path)

        all_dets: list = []
        redacted_pages: list = []

        for page_no, page in enumerate(pages, start=1):
            dets = self._detector.detect(page, conf=conf)
            if run_ocr:
                for d in dets:
                    d["ocr_text"] = extract_text(page, d["bbox"])
            for d in dets:
                d["page"] = page_no
            all_dets.extend(dets)
            redacted_pages.append(redact_image(page, dets))

        is_pdf = doc_path.suffix.lower() == ".pdf"
        if is_pdf:
            out_path = output_dir / f"{stem}_redacted_{ts}.pdf"
            redacted_pages[0].save(
                out_path,
                format="PDF",
                save_all=True,
                append_images=redacted_pages[1:],
            )
        else:
            out_path = output_dir / f"{stem}_redacted_{ts}.png"
            redacted_pages[0].save(out_path)

        log_path = output_dir / f"{stem}_audit_{ts}.json"
        audit = write_audit_log(doc_path, all_dets, out_path, log_path)

        return redacted_pages, audit
