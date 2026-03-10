"""
agent.py — Plan → Retrieve → Extract → Verify orchestration loop.

The agent exposes a run() method that accepts a status_callback so Streamlit
can display live progress messages without coupling to Streamlit internals.
"""

from __future__ import annotations
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from paper_agent import prompts
from paper_agent.schemas import (
    PaperSummary, PaperAnalysis, KeyFinding, ResearchGap,
    MethodItem, FutureWorkItem, Evidence,
)
from paper_agent.utils import safe_parse_json

logger = logging.getLogger(__name__)

# Minimum evidence items before triggering a second retrieval pass
_MIN_EVIDENCE = 2

StatusFn = Callable[[str], None]


class PaperAgent:
    """
    Orchestrates the full summarisation pipeline.

    Parameters
    ----------
    store:            FAISSVectorStore (or raw FAISS store with similarity_search).
    first_page_text:  Text from first 1–3 pages (for planning).
    chat_fn:          Callable(user_msg) → str  (from llm.make_chat_fn).
    top_k:            Chunks to retrieve per query.
    status_fn:        Optional callback(str) for live status updates.
    """

    def __init__(
        self,
        store,
        first_page_text: str,
        chat_fn: Callable[[str], str],
        top_k: int = 6,
        status_fn: Optional[StatusFn] = None,
    ):
        self.store = store
        self.first_page_text = first_page_text
        self.chat = chat_fn
        self.top_k = top_k
        self._status = status_fn or (lambda msg: logger.info(msg))

    # ── Status helper ─────────────────────────────────────────────────────────

    def _s(self, msg: str):
        self._status(msg)
        logger.info(msg)

    # ── LLM call wrapper ──────────────────────────────────────────────────────

    def _llm(self, prompt: str, label: str) -> Any:
        self._s(f"  ↳ LLM: {label}")
        raw = self.chat(prompt)
        try:
            return safe_parse_json(raw)
        except ValueError as exc:
            logger.warning("JSON parse failed for %s: %s", label, exc)
            return None

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _retrieve(self, queries: List[str]) -> str:
        from paper_agent.vectorstore import retrieve_context
        # FAISSVectorStore wraps the raw FAISS store; retrieve_context needs raw store
        raw_store = getattr(self.store, "store", self.store)
        return retrieve_context(raw_store, queries, k=self.top_k)

    # ── Step 1: PLAN ──────────────────────────────────────────────────────────

    def _plan(self) -> Dict:
        self._s("📋 Step 1/4 — Planning retrieval queries…")
        raw = self.chat(prompts.PLANNING_PROMPT.format(first_page_text=self.first_page_text))
        try:
            plan = safe_parse_json(raw)
        except ValueError:
            logger.warning("Planning JSON failed; using defaults.")
            plan = {}

        defaults = {
            "key_findings": ["main results", "key contributions", "findings"],
            "research_gap":  ["limitations", "research gap", "future challenges"],
            "methods_used":  ["methodology", "model", "approach", "algorithm"],
            "future_work":   ["future work", "open problems", "extensions"],
        }
        queries = plan.get("queries") or {}
        for k, v in defaults.items():
            if k not in queries or not queries[k]:
                queries[k] = v
        plan["queries"] = queries
        return plan

    # ── Step 2 & 3: RETRIEVE + EXTRACT ────────────────────────────────────────

    def _extract_all(self, plan: Dict) -> Dict:
        self._s("🔍 Step 2/4 — Retrieving passages…")
        q = plan["queries"]

        ctx_findings = self._retrieve(q["key_findings"])
        ctx_gap      = self._retrieve(q["research_gap"])
        ctx_methods  = self._retrieve(q["methods_used"])
        ctx_future   = self._retrieve(q["future_work"])
        ctx_broad    = self._retrieve(q["key_findings"] + q["methods_used"] + q["future_work"])

        self._s("✍️  Step 3/4 — Generating sections…")

        def _list(result, label):
            if isinstance(result, list):
                return result
            logger.warning("%s returned non-list; defaulting to [].", label)
            return []

        findings = _list(self._llm(prompts.KEY_FINDINGS_PROMPT.format(context=ctx_findings), "key_findings"), "key_findings")
        gap      = _list(self._llm(prompts.RESEARCH_GAP_PROMPT.format(context=ctx_gap),      "research_gap"),  "research_gap")
        methods  = _list(self._llm(prompts.METHODS_PROMPT.format(context=ctx_methods),        "methods_used"),  "methods_used")
        future   = _list(self._llm(prompts.FUTURE_WORK_PROMPT.format(context=ctx_future),     "future_work"),   "future_work")
        tldr     = _list(self._llm(prompts.TLDR_PROMPT.format(context=ctx_broad),             "tldr"),          "tldr")
        limits   = _list(self._llm(prompts.LIMITATIONS_PROMPT.format(context=ctx_broad),      "limitations"),   "limitations")

        return {
            "title":         plan.get("title", "Unknown Title"),
            "authors":       plan.get("authors") or [],
            "year":          plan.get("year"),
            "key_findings":  findings,
            "research_gap":  gap,
            "methods_used":  methods,
            "future_work":   future,
            "tldr_bullets":  [str(b) for b in tldr],
            "limitations":   [str(l) for l in limits],
        }

    # ── Step 4: VERIFY ────────────────────────────────────────────────────────

    def _verify(self, data: Dict, plan: Dict) -> Dict:
        self._s("🔎 Step 4/4 — Verifying evidence depth…")
        q = plan["queries"]

        section_cfg = {
            "key_findings": (
                prompts.KEY_FINDINGS_PROMPT,
                q["key_findings"] + ["results", "evaluation", "performance", "accuracy"],
            ),
            "research_gap": (
                prompts.RESEARCH_GAP_PROMPT,
                q["research_gap"] + ["limitation", "drawback", "weakness", "constraint"],
            ),
            "methods_used": (
                prompts.METHODS_PROMPT,
                q["methods_used"] + ["dataset", "implementation", "training", "experiment"],
            ),
            "future_work": (
                prompts.FUTURE_WORK_PROMPT,
                q["future_work"] + ["open questions", "recommend", "next steps"],
            ),
        }

        for section, (prompt_tpl, broader_queries) in section_cfg.items():
            current = data.get(section, [])
            min_ev = min((len(item.get("evidence") or []) for item in current), default=0)
            if min_ev < _MIN_EVIDENCE:
                self._s(f"  ↳ Low evidence in '{section}' — running second pass…")
                ctx2 = self._retrieve(broader_queries)
                refined = self._llm(prompt_tpl.format(context=ctx2), f"{section}_pass2")
                if isinstance(refined, list) and refined:
                    data[section] = refined
        return data

    # ── Assemble PaperSummary Pydantic model ──────────────────────────────────

    @staticmethod
    def _to_pydantic(data: Dict) -> PaperSummary:
        def ev(lst):
            out = []
            for e in (lst or []):
                try:
                    out.append(Evidence(page=int(e.get("page", 0)), quote=str(e.get("quote", ""))))
                except Exception:
                    pass
            return out

        tldr = data.get("tldr_bullets") or []
        tldr = tldr[:12]
        while len(tldr) < 8:
            tldr.append("(Not found in retrieved passages.)")

        return PaperSummary(
            title=data.get("title", "Unknown Title"),
            authors=data.get("authors") or [],
            year=data.get("year"),
            tldr_bullets=tldr,
            key_findings=[KeyFinding(claim=f.get("claim", ""), evidence=ev(f.get("evidence"))) for f in data.get("key_findings") or []],
            research_gap=[ResearchGap(gap=g.get("gap", ""), evidence=ev(g.get("evidence"))) for g in data.get("research_gap") or []],
            methods_used=[MethodItem(method=m.get("method", ""), details=m.get("details", ""), evidence=ev(m.get("evidence"))) for m in data.get("methods_used") or []],
            future_work=[FutureWorkItem(item=fw.get("item", ""), evidence=ev(fw.get("evidence"))) for fw in data.get("future_work") or []],
            limitations=data.get("limitations") or [],
            confidence_notes=data.get("confidence_notes", ""),
        )

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self) -> PaperSummary:
        plan = self._plan()
        data = self._extract_all(plan)
        data = self._verify(data, plan)

        # Confidence
        self._s("📊 Generating confidence assessment…")
        raw_conf = self.chat(prompts.CONFIDENCE_PROMPT.format(
            extraction_json=json.dumps(data, indent=2)[:3000]
        ))
        try:
            conf = safe_parse_json(raw_conf)
            data["confidence_notes"] = str(conf) if isinstance(conf, str) else json.dumps(conf)
        except Exception:
            data["confidence_notes"] = raw_conf[:500]

        return self._to_pydantic(data)


# ── Streamlit-facing wrapper ──────────────────────────────────────────────────

def analyze_single_paper(
    vs,
    paper_name: str,
    provider: str,
    model: str,
    top_k: int,
    progress_cb=None,
) -> PaperAnalysis:
    """
    Wrapper used by the Streamlit app.

    Creates the PaperAgent, runs the full pipeline, and converts the result
    to a flat PaperAnalysis ready for display.
    """
    from paper_agent.llm import make_chat_fn

    chat_fn = make_chat_fn(provider, model, None)

    # Get first-page text via vector similarity search
    docs = vs.similarity_search(paper_name, k=2)
    first_page_text = "\n".join([d.page_content for d in docs])

    agent = PaperAgent(
        store=vs,
        first_page_text=first_page_text,
        chat_fn=chat_fn,
        top_k=top_k,
        status_fn=progress_cb,
    )

    summary: PaperSummary = agent.run()
    return PaperAnalysis.from_summary(summary)
