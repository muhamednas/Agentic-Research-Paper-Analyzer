"""
pdf_loader.py — Extract text from PDF with page-number metadata preserved.

Returns: List[{"page": int, "text": str}]

Accepts either:
  - load_pdf_pages(pdf_path: Path)              — original file-path API
  - load_pdf_pages(pdf_bytes: bytes, fname: str) — bytes API used by Streamlit app

Handles:
  - Encrypted PDFs (attempts blank-password decrypt)
  - Empty pages (skipped with warning)
  - Per-page extraction failures (skipped individually)
  - Scanned-image PDFs (raises clear error)
"""

from __future__ import annotations
import hashlib
import io
import logging
import re
from pathlib import Path
from typing import List, Dict, Union

logger = logging.getLogger(__name__)


def load_pdf_pages(
    source: Union[Path, bytes],
    fname: str = "document.pdf",
) -> List[Dict]:
    """
    Extract text from every page, returning list of {page, text} dicts.

    Parameters
    ----------
    source : Path | bytes
        Either a filesystem path or raw PDF bytes (from st.file_uploader).
    fname  : str
        Display name used only for log messages when source is bytes.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf is not installed. Run: pip install pypdf")

    # Support both Path and raw bytes
    if isinstance(source, (str, Path)):
        reader = PdfReader(str(source))
        display_name = Path(source).name
    elif isinstance(source, bytes):
        reader = PdfReader(io.BytesIO(source))
        display_name = fname
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise RuntimeError(f"PDF is password-protected: {display_name}")

    total = len(reader.pages)
    if total == 0:
        raise ValueError("PDF contains no pages.")

    pages: List[Dict] = []
    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        try:
            raw = page.extract_text() or ""
        except Exception as exc:
            logger.warning("Page %d extraction failed: %s — skipping.", page_num, exc)
            continue

        text = _clean(raw)
        if text.strip():
            pages.append({"page": page_num, "text": text})

    if not pages:
        raise ValueError(
            "No text could be extracted. This may be a scanned image PDF. "
            "Please run OCR (e.g. ocrmypdf) before uploading."
        )

    logger.info("Extracted %d/%d pages from %s", len(pages), total, display_name)
    return pages


def get_first_page_text(pages: List[Dict], max_chars: int = 3000) -> str:
    """Return combined text of first ~3 pages, trimmed to max_chars."""
    combined = "\n\n".join(p["text"] for p in pages[:3])
    return combined[:max_chars]


def pdf_hash(pdf_bytes: bytes) -> str:
    """SHA-256 hex digest of raw PDF bytes (used for cache keying)."""
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]


# ── Internal ──────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    text = re.sub(r"[\f\r]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
