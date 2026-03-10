"""
vectorstore.py — Build, persist, and query a FAISS vector store.

Index is cached to .cache/<hash>/ for fast reloads.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

# Default embedding model (used by app.py sidebar display and FAISSVectorStore)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings(provider: str, embed_model: str = EMBEDDING_MODEL_NAME, api_key: str = "", api_base: str | None = None):
    """Return a LangChain Embeddings object for the selected provider."""

    if embed_model.startswith("sentence-transformers") or embed_model == "all-MiniLM-L6-v2":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        model_name = embed_model
        if embed_model == "all-MiniLM-L6-v2":
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        return HuggingFaceEmbeddings(model_name=model_name)

    if provider in ("OpenAI-compatible",):
        from langchain_openai import OpenAIEmbeddings
        kwargs = {"model": embed_model, "api_key": api_key}
        if api_base:
            kwargs["base_url"] = api_base
        return OpenAIEmbeddings(**kwargs)

    elif provider == "Gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=embed_model, google_api_key=api_key)

    # Groq, Anthropic, HuggingFace — all fall back to local sentence-transformers
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_vectorstore(docs: List[Document], index_dir: Path, embeddings) -> FAISS:
    """Embed docs and save FAISS index to index_dir."""
    if not docs:
        raise ValueError("Cannot build vector store from zero documents.")
    logger.info("Building FAISS index from %d chunks…", len(docs))
    store = FAISS.from_documents(docs, embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir))
    logger.info("FAISS index saved → %s", index_dir)
    return store


def load_vectorstore(index_dir: Path, embeddings) -> FAISS:
    """Load a previously saved FAISS index."""
    if not index_dir.exists():
        raise FileNotFoundError(f"No FAISS index at {index_dir}")
    logger.info("Loading FAISS index from %s", index_dir)
    return FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)


def get_or_build(
    docs: List[Document],
    index_dir: Path,
    embeddings,
    rebuild: bool = False,
) -> FAISS:
    """Load existing index or build a new one."""
    if not rebuild and index_dir.exists():
        try:
            return load_vectorstore(index_dir, embeddings)
        except Exception as exc:
            logger.warning("Could not load existing index (%s); rebuilding.", exc)
    return build_vectorstore(docs, index_dir, embeddings)


def retrieve_context(store: FAISS, queries: List[str], k: int = 6) -> str:
    """
    Run multiple queries, deduplicate by chunk_id, return formatted context string.
    Format: [Page N | Chunk id]\n<text>
    """
    seen: set[str] = set()
    collected: List[Document] = []

    for query in queries:
        for doc, _ in store.similarity_search_with_score(query, k=k):
            cid = doc.metadata.get("chunk_id", "?")
            if cid not in seen:
                seen.add(cid)
                collected.append(doc)

    if not collected:
        return "(No relevant passages found.)"

    parts = []
    for doc in collected:
        page = doc.metadata.get("page", "?")
        cid = doc.metadata.get("chunk_id", "?")
        parts.append(f"[Page {page} | Chunk {cid}]\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)


# ─────────────────────────────────────────────────────────────
# Streamlit wrapper class used by app.py
# ─────────────────────────────────────────────────────────────

class FAISSVectorStore:
    """
    Thin wrapper around FAISS helper functions so the Streamlit app
    can call build(), load(), save(), and search().
    """

    def __init__(self):
        self.store: FAISS | None = None
        self._embeddings = None

    def _get_default_embeddings(self):
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    def build(self, docs: List[Document], embeddings=None):
        if embeddings is None:
            embeddings = self._get_default_embeddings()
        self._embeddings = embeddings
        self.store = FAISS.from_documents(docs, embeddings)

    def save(self, index_dir):
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        if self.store is None:
            raise ValueError("Vector store is empty.")
        self.store.save_local(str(index_dir))

    def load(self, index_dir) -> bool:
        index_dir = Path(index_dir)
        if not index_dir.exists():
            return False
        try:
            embeddings = self._get_default_embeddings()
            self._embeddings = embeddings
            self.store = FAISS.load_local(
                str(index_dir),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            return True
        except Exception as exc:
            logger.warning("Failed to load FAISS index: %s", exc)
            return False

    def search(self, query: str, k: int = 6) -> List[Document]:
        if self.store is None:
            raise ValueError("Vector store not initialized.")
        return self.store.similarity_search(query, k=k)

    def similarity_search(self, query: str, k: int = 6) -> List[Document]:
        """Alias so agent code using store.similarity_search() works too."""
        return self.search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 6):
        if self.store is None:
            raise ValueError("Vector store not initialized.")
        return self.store.similarity_search_with_score(query, k=k)
