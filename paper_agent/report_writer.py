"""
report_writer.py — Convert a PaperSummary/PaperAnalysis into Markdown / JSON,
and generate a cross-paper LiteratureReview.
"""

from __future__ import annotations
import json
import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def to_markdown(summary: Dict[str, Any]) -> str:
    """Render the structured summary dict as a readable Markdown report."""
    lines: list[str] = []

    def h(level: int, text: str):
        lines.append(f"\n{'#' * level} {text}\n")

    def bullet(text: str):
        lines.append(f"- {text}")

    def evidence_block(ev_list: list):
        for ev in ev_list or []:
            page = ev.get("page", "?")
            quote = ev.get("quote", "").strip()
            if quote:
                lines.append(f'  > *p.{page}* — "{quote}"')

    # Header
    title = summary.get("title", "Untitled")
    authors = ", ".join(summary.get("authors") or []) or "Unknown"
    year = summary.get("year") or "n.d."
    lines.append(f"# {title}")
    lines.append(f"\n**Authors:** {authors}  \n**Year:** {year}\n")
    lines.append("---\n")

    h(2, "TL;DR")
    for b in summary.get("tldr_bullets") or []:
        bullet(b)

    h(2, "Key Findings")
    findings = summary.get("key_findings") or []
    if findings:
        for i, f in enumerate(findings, 1):
            lines.append(f"\n**{i}. {f.get('claim', '')}**")
            evidence_block(f.get("evidence"))
    else:
        lines.append("_Not found in paper._")

    h(2, "Research Gap")
    gaps = summary.get("research_gap") or []
    if gaps:
        for i, g in enumerate(gaps, 1):
            lines.append(f"\n**{i}. {g.get('gap', '')}**")
            evidence_block(g.get("evidence"))
    else:
        lines.append("_Not found in paper._")

    h(2, "Methods Used")
    methods = summary.get("methods_used") or []
    if methods:
        for m in methods:
            lines.append(f"\n**{m.get('method', '')}**")
            lines.append(f"{m.get('details', '')}")
            evidence_block(m.get("evidence"))
    else:
        lines.append("_Not found in paper._")

    h(2, "Future Work")
    future = summary.get("future_work") or []
    if future:
        for fw in future:
            bullet(fw.get("item", ""))
            evidence_block(fw.get("evidence"))
    else:
        lines.append("_Not found in paper._")

    h(2, "Limitations")
    limits = summary.get("limitations") or []
    if limits:
        for lim in limits:
            bullet(lim)
    else:
        lines.append("_Not found in paper._")

    h(2, "Agent Confidence Notes")
    lines.append(summary.get("confidence_notes") or "_No notes._")

    return "\n".join(lines) + "\n"


def to_json_str(summary: Dict[str, Any]) -> str:
    return json.dumps(summary, indent=2, ensure_ascii=False)


# ── Cross-paper literature review ─────────────────────────────────────────────

def generate_literature_review(
    analyses,          # List[PaperAnalysis]
    provider: str,
    model: str,
    api_key: str | None = None,
):
    """
    Synthesise multiple PaperAnalysis objects into a LiteratureReview.

    Uses the same LLM factory as the rest of the app.
    Falls back to a plain-text concatenation if the LLM call fails.
    """
    from paper_agent.schemas import LiteratureReview
    from paper_agent.llm import make_chat_fn
    from paper_agent.utils import safe_parse_json

    chat_fn = make_chat_fn(provider, model, api_key)

    # Build a compact digest of all analyses to send to the LLM
    digests = []
    for a in analyses:
        digests.append(
            f"=== {a.title} ===\n"
            f"TL;DR:\n{a.tldr}\n\n"
            f"Key Findings:\n{a.key_findings}\n\n"
            f"Methods:\n{a.methods_used}\n\n"
            f"Research Gap:\n{a.research_gap}\n"
        )
    combined = "\n\n".join(digests)

    prompt = f"""You are given summaries of {len(analyses)} research papers.
Write a structured literature review as a JSON object with these keys:
  - "overview":          2–4 sentence overview of all papers together
  - "method_comparison": comparison of methods used across papers
  - "common_findings":   findings or themes that appear in multiple papers
  - "cross_paper_gaps":  research gaps common across papers

Use ONLY the information provided. Return pure JSON only.

Paper summaries:
---
{combined[:6000]}
---
"""

    try:
        raw = chat_fn(prompt)
        data = safe_parse_json(raw)
        return LiteratureReview(
            overview=data.get("overview", ""),
            method_comparison=data.get("method_comparison", ""),
            common_findings=data.get("common_findings", ""),
            cross_paper_gaps=data.get("cross_paper_gaps", ""),
        )
    except Exception as exc:
        logger.warning("Literature review LLM call failed: %s", exc)
        # Graceful fallback
        from paper_agent.schemas import LiteratureReview
        return LiteratureReview(
            overview=f"Review of {len(analyses)} papers. LLM synthesis unavailable.",
            method_comparison="See individual paper analyses above.",
            common_findings="See individual paper analyses above.",
            cross_paper_gaps="See individual paper analyses above.",
        )
