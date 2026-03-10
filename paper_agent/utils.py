"""
utils.py — JSON parsing helpers and misc utilities.
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any


def safe_parse_json(text: str) -> Any:
    """
    Parse JSON from (possibly noisy) LLM output.
    Strips markdown fences before parsing.
    Raises ValueError with context on failure.
    """
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned non-parseable JSON.\n"
            f"Preview: {text[:300]}\nError: {exc}"
        ) from exc


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def cache_dir_for_hash(pdf_hash: str) -> Path:
    """Return a Path for a cache directory keyed by pdf_hash."""
    return Path(".cache") / pdf_hash / "faiss_index"


def get_cache_dir(cache_key: str) -> Path:
    """
    Return a Path for a cache directory keyed by cache_key.
    Used by app.py to persist / reload FAISS indexes.
    """
    cache_path = Path(".cache") / cache_key
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path
