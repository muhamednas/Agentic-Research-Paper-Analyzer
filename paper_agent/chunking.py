"""
chunking.py — Split per-page texts into overlapping chunks, preserving page metadata.

Each LangChain Document produced carries:
  metadata = {"page": int, "chunk_id": "page-localIdx", "source": <fname>}

Signature supports both:
  chunk_pages(pages)                              — original API
  chunk_pages(pages, fname, chunk_size, overlap)  — extended API used by app.py
"""

from __future__ import annotations
import logging
from typing import List, Dict

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def chunk_pages(
    pages: List[Dict],
    source: str = "paper",
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Convert {page, text} dicts into LangChain Documents with metadata.

    Parameters
    ----------
    pages:         output of pdf_loader.load_pdf_pages()
    source:        label stored in metadata["source"] (e.g. the filename)
    chunk_size:    target character length per chunk
    chunk_overlap: character overlap between adjacent chunks
    """
    if not pages:
        raise ValueError("No pages to chunk.")

    splitter = RecursiveCharacterTextSplitter(
        separators=_SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    docs: List[Document] = []
    for p in pages:
        page_num = p["page"]
        chunks = splitter.split_text(p["text"])
        for local_idx, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if chunk:
                docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "page": page_num,
                        "chunk_id": f"{page_num}-{local_idx}",
                        "source": source,
                    },
                ))

    if not docs:
        raise ValueError("Chunking produced zero documents.")

    logger.info(
        "Chunked %d pages → %d chunks (size=%d, overlap=%d, source=%s)",
        len(pages), len(docs), chunk_size, chunk_overlap, source,
    )
    return docs
