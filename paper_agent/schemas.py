"""
schemas.py — Pydantic v2 models for the structured paper extraction output.
Every claim carries at least one Evidence atom (page + quote).
"""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


# ── Low-level evidence atom ───────────────────────────────────────────────────

class Evidence(BaseModel):
    page: int = Field(..., description="1-based page number in the PDF")
    quote: str = Field(..., description="Short verbatim or near-verbatim excerpt (≤200 chars)")


# ── Section-level models ──────────────────────────────────────────────────────

class KeyFinding(BaseModel):
    claim: str
    evidence: List[Evidence] = Field(default_factory=list)


class ResearchGap(BaseModel):
    gap: str
    evidence: List[Evidence] = Field(default_factory=list)


class MethodItem(BaseModel):
    method: str
    details: str
    evidence: List[Evidence] = Field(default_factory=list)


class FutureWorkItem(BaseModel):
    item: str
    evidence: List[Evidence] = Field(default_factory=list)


# ── Full paper summary (internal / legacy) ────────────────────────────────────

class PaperSummary(BaseModel):
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(default_factory=list)
    year: Optional[str] = None
    tldr_bullets: List[str] = Field(default_factory=list)
    key_findings: List[KeyFinding] = Field(default_factory=list)
    research_gap: List[ResearchGap] = Field(default_factory=list)
    methods_used: List[MethodItem] = Field(default_factory=list)
    future_work: List[FutureWorkItem] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    confidence_notes: str = ""


# ── Flat analysis model consumed by app.py ────────────────────────────────────

class PaperAnalysis(BaseModel):
    """
    Flat, display-friendly model returned by analyze_single_paper()
    and consumed directly by the Streamlit UI.
    """
    title: str = ""
    authors: List[str] = Field(default_factory=list)
    year: Optional[str] = None

    # Plain-text fields rendered in result_box() cards
    tldr: str = ""
    key_findings: str = ""
    research_gap: str = ""
    methods_used: str = ""
    future_work: str = ""
    limitations: str = ""
    confidence_notes: str = ""

    @classmethod
    def from_summary(cls, summary: PaperSummary) -> "PaperAnalysis":
        """Convert a rich PaperSummary into the flat PaperAnalysis."""

        def fmt_findings(items: List[KeyFinding]) -> str:
            if not items:
                return "Not found in paper."
            lines = []
            for i, f in enumerate(items, 1):
                lines.append(f"{i}. {f.claim}")
                for ev in f.evidence:
                    lines.append(f'   › p.{ev.page} — "{ev.quote[:120]}"')
            return "\n".join(lines)

        def fmt_gaps(items: List[ResearchGap]) -> str:
            if not items:
                return "Not found in paper."
            lines = []
            for i, g in enumerate(items, 1):
                lines.append(f"{i}. {g.gap}")
                for ev in g.evidence:
                    lines.append(f'   › p.{ev.page} — "{ev.quote[:120]}"')
            return "\n".join(lines)

        def fmt_methods(items: List[MethodItem]) -> str:
            if not items:
                return "Not found in paper."
            lines = []
            for m in items:
                lines.append(f"• {m.method}: {m.details}")
                for ev in m.evidence:
                    lines.append(f'   › p.{ev.page} — "{ev.quote[:120]}"')
            return "\n".join(lines)

        def fmt_future(items: List[FutureWorkItem]) -> str:
            if not items:
                return "Not found in paper."
            lines = []
            for fw in items:
                lines.append(f"• {fw.item}")
                for ev in fw.evidence:
                    lines.append(f'   › p.{ev.page} — "{ev.quote[:120]}"')
            return "\n".join(lines)

        tldr_text = "\n".join(
            f"• {b}" for b in summary.tldr_bullets if b and b != "(Not found in retrieved passages.)"
        ) or "Not found in paper."

        limits_text = (
            "\n".join(f"• {l}" for l in summary.limitations)
            if summary.limitations
            else "Not found in paper."
        )

        return cls(
            title=summary.title,
            authors=summary.authors,
            year=summary.year,
            tldr=tldr_text,
            key_findings=fmt_findings(summary.key_findings),
            research_gap=fmt_gaps(summary.research_gap),
            methods_used=fmt_methods(summary.methods_used),
            future_work=fmt_future(summary.future_work),
            limitations=limits_text,
            confidence_notes=summary.confidence_notes,
        )


# ── Cross-paper literature review ─────────────────────────────────────────────

class LiteratureReview(BaseModel):
    """Returned by generate_literature_review() in report_writer.py."""
    overview: str = ""
    method_comparison: str = ""
    common_findings: str = ""
    cross_paper_gaps: str = ""
