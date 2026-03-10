"""
prompts.py — All LLM prompts used by the agent.

Design rules:
  - Every section prompt demands page-number citations + short quotes.
  - "Not found in paper" is the required response when evidence is absent.
  - Pure JSON output only — no markdown fences, no prose preamble.
"""

# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a rigorous academic research assistant with zero tolerance for hallucination.

ABSOLUTE RULES:
1. GROUND every claim in the retrieved passages. For each piece of evidence provide:
   - "page": integer (1-based page number, exactly as shown in the context header)
   - "quote": ≤200-character verbatim or near-verbatim excerpt from that page
2. NEVER invent page numbers, authors, quotes, or results not present in the context.
3. If information is absent from the provided passages, write exactly: "Not found in paper"
4. Return ONLY a valid JSON object or JSON array — no markdown fences, no extra text.
5. Prefer 3–6 well-evidenced items per section over many unsupported ones.
"""

# ── Planning prompt ──────────────────────────────────────────────────────────

PLANNING_PROMPT = """Analyze the first-page text of a research paper and return a JSON object with:
  - "title": string
  - "authors": list of strings (empty list if not found)
  - "year": string or null
  - "queries": object with keys:
      "key_findings"  → list of 3 search query strings
      "research_gap"  → list of 3 search query strings
      "methods_used"  → list of 3 search query strings
      "future_work"   → list of 3 search query strings

First-page text:
---
{first_page_text}
---

Return pure JSON only.
"""

# ── Section prompts ──────────────────────────────────────────────────────────

KEY_FINDINGS_PROMPT = """Extract KEY FINDINGS from the retrieved passages below.

For each finding:
  - "claim": one concise sentence stating the finding
  - "evidence": list of {{"page": <int>, "quote": "<excerpt ≤200 chars>"}}

If no findings are found: return []

Retrieved passages:
---
{context}
---

Return a JSON array only.
"""

RESEARCH_GAP_PROMPT = """Extract RESEARCH GAPS or LIMITATIONS acknowledged by the authors.
These are problems left unsolved, or constraints of their own approach.

For each gap:
  - "gap": concise description
  - "evidence": list of {{"page": <int>, "quote": "<excerpt ≤200 chars>"}}

If none found: return []

Retrieved passages:
---
{context}
---

Return a JSON array only.
"""

METHODS_PROMPT = """Extract METHODS and TECHNIQUES used in this paper.

For each method:
  - "method": name
  - "details": 1–3 sentence explanation of its role
  - "evidence": list of {{"page": <int>, "quote": "<excerpt ≤200 chars>"}}

If none found: return []

Retrieved passages:
---
{context}
---

Return a JSON array only.
"""

FUTURE_WORK_PROMPT = """Extract FUTURE WORK directions explicitly mentioned by the authors.

For each direction:
  - "item": concise description
  - "evidence": list of {{"page": <int>, "quote": "<excerpt ≤200 chars>"}}

If none found: return []

Retrieved passages:
---
{context}
---

Return a JSON array only.
"""

TLDR_PROMPT = """Write a TL;DR summary as exactly 8–12 concise bullet-point sentences.
Each bullet must be self-contained and informative.
Use ONLY information present in the retrieved passages.

Retrieved passages:
---
{context}
---

Return a JSON array of strings (the bullet points). No bullet symbols.
"""

LIMITATIONS_PROMPT = """List LIMITATIONS of this paper (stated or clearly implied).

Retrieved passages:
---
{context}
---

Return a JSON array of strings. If none found: return []
"""

CONFIDENCE_PROMPT = """Given the structured extraction below, write a 2–4 sentence
self-assessment of extraction confidence:
- Which sections are well-supported?
- Which have sparse or missing evidence?
- Any notable caveats?

Extraction:
---
{extraction_json}
---

Return a single JSON string (just the string value, in double quotes).
"""
