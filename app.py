import os
import hashlib
from typing import List

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from paper_agent.pdf_loader import load_pdf_pages
from paper_agent.chunking import chunk_pages
from paper_agent.vectorstore import FAISSVectorStore, EMBEDDING_MODEL_NAME
from paper_agent.agent import analyze_single_paper
from paper_agent.report_writer import generate_literature_review
from paper_agent.utils import get_cache_dir
from paper_agent.schemas import PaperAnalysis

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic Research Paper Analyzer",
    page_icon=" ",
    layout="wide",
)

# ─── Global Styles (CSS) ────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* App background + typography */
.stApp {
    background: radial-gradient(1200px 800px at 20% 10%, rgba(108,99,255,0.12), transparent 60%),
                radial-gradient(1000px 700px at 90% 20%, rgba(0,200,255,0.10), transparent 55%),
                linear-gradient(180deg, #0b0b14 0%, #0b0b14 100%);
    color: #f2f2f7;
}
h1, h2, h3 { letter-spacing: 0.2px; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(18,18,32,0.95), rgba(12,12,22,0.95));
    border-right: 1px solid rgba(255,255,255,0.06);
}
.stButton>button {
    border: 0;
    border-radius: 14px;
    padding: 0.85rem 1.1rem;
    font-weight: 700;
    letter-spacing: 0.2px;
    background: linear-gradient(90deg, #6c63ff 0%, #00d4ff 100%);
    color: #0b0b14;
    box-shadow: 0 10px 30px rgba(108,99,255,0.25);
    transition: transform 0.08s ease, filter 0.12s ease;
}
.stButton>button:hover  { transform: translateY(-1px); filter: brightness(1.05); }
.stButton>button:active { transform: translateY(0px);  filter: brightness(0.98); }
div[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    background: rgba(255,255,255,0.03);
}
.card {
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 1rem 1rem 0.8rem;
    background: rgba(255,255,255,0.03);
    box-shadow: 0 10px 25px rgba(0,0,0,0.25);
}
.card h4 { margin: 0 0 0.6rem 0; font-size: 1.02rem; }
.mini    { opacity: 0.85; font-size: 0.90rem; line-height: 1.4; }
.badge {
    display: inline-block;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    background: rgba(108,99,255,0.18);
    border: 1px solid rgba(108,99,255,0.35);
    color: #d8d6ff;
    margin-left: 0.4rem;
}
hr { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 1.0rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

# ─── Helpers ────────────────────────────────────────────────────────────────

def combined_upload_hash(files: list) -> str:
    """Stable combined hash across multiple uploaded PDFs without consuming file pointers."""
    h = hashlib.sha256()
    for f in files:
        b = f.getvalue()
        h.update(f.name.encode("utf-8", errors="ignore"))
        h.update(len(b).to_bytes(8, "big", signed=False))
        h.update(hashlib.sha256(b).digest())
    return h.hexdigest()


def card(title: str, body, help_text: str = ""):
    """Render a consistent output card."""
    st.markdown(f"<div class='card'><h4>{title}</h4>", unsafe_allow_html=True)
    if help_text:
        st.markdown(f"<div class='mini'>{help_text}</div><hr/>", unsafe_allow_html=True)
    st.write(body)
    st.markdown("</div>", unsafe_allow_html=True)


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown(
        "<div class='mini'>Choose which AI model will <b>read</b> and <b>explain</b> your PDFs. "
        "Embeddings are fixed to keep the system stable.</div>",
        unsafe_allow_html=True,
    )

    st.subheader("AI Provider (LLM)")
    provider = st.selectbox("Provider", ["groq", "huggingface"], index=0)

    st.subheader("AI Model")
    if provider == "groq":
        model_name = st.selectbox(
            "Chat Model",
            [
                "meta-llama/llama-4-maverick-17b-128e-instruct",
                "meta-llama/llama-4-scout-17b-16e-instruct",
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
            ],
            index=0,
        )
    else:
        model_name = st.selectbox(
            "Chat Model",
            [
                "meta-llama/Meta-Llama-3-8B-Instruct",
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "google/gemma-7b-it",
            ],
            index=0,
        )

    st.subheader("Embeddings (Fixed)")
    st.info(f"🔒 `{EMBEDDING_MODEL_NAME}`")

    with st.expander("What do these words mean? (Quick guide)", expanded=False):
        st.markdown(
            """
**LLM (Large Language Model):** the AI "brain" that reads text and writes explanations.

**Embeddings:** convert text into numbers so the computer can find *similar* meaning fast.

**Vector Database (FAISS):** a fast search system that uses embeddings to find the best parts of the paper.

**Chunking:** splitting a long PDF into smaller parts so search and analysis becomes easier.

**Top-K:** how many best-matching chunks we pull from the paper for the AI to read.
"""
        )

    st.subheader("How the PDF is split (Chunking)")
    chunk_size    = st.slider("Chunk Size (characters)",    500, 3000, 1200, 100)
    chunk_overlap = st.slider("Chunk Overlap (characters)",   0,  500,  200,  50)

    st.subheader("Search depth (Retrieval)")
    top_k = st.slider("Top results per question (Top-K)", 2, 20, 6)

    st.subheader("Index (Cache)")
    rebuild_index = st.checkbox("  Rebuild search index", value=False)

    st.divider()
    st.caption("API keys loaded from `.env` file")
    groq_key_set = bool(os.getenv("GROQ_API_KEY"))
    hf_key_set   = bool(os.getenv("HUGGINGFACE_API_KEY"))
   
# ─── Main UI ────────────────────────────────────────────────────────────────
st.title("  Agentic Research Paper Analyzer")
st.markdown(
    """
Upload one or more academic PDFs.  
This tool will **read the papers** and generate **easy-to-understand insights** like:  
**Summary, Key Findings, Methods, Research Gaps, Limitations, and Future Work**.  
If you upload multiple papers, it also creates a **Literature Review** (a comparison across papers).
"""
)

uploaded_files = st.file_uploader(
    "Upload PDF(s)",
    type=["pdf"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info(" Upload one or more research papers to begin.")
    st.stop()

st.markdown(
    "<div class='mini'>Tip: Start with 1 paper, then try 2–3 papers for the literature review.</div>",
    unsafe_allow_html=True,
)

# ─── Run ────────────────────────────────────────────────────────────────────
if st.button("  Analyze Papers", type="primary", use_container_width=True):

    # API key gate
    if provider == "groq" and not groq_key_set:
        st.error("  GROQ_API_KEY is missing in your `.env`. Add it and restart Streamlit.")
        st.stop()
    if provider == "huggingface" and not hf_key_set:
        st.error("  HUGGINGFACE_API_KEY is missing in your `.env`. Add it and restart Streamlit.")
        st.stop()

    status_box = st.empty()
    all_analyses: List[PaperAnalysis] = []

    status_box.info(" Preparing the smart search index...")

    files_hash = combined_upload_hash(uploaded_files)
    safe_embed = EMBEDDING_MODEL_NAME.replace("/", "_").replace(":", "_").replace("\\", "_")
    cache_key  = f"{files_hash}_emb_{safe_embed}_cs_{chunk_size}_co_{chunk_overlap}"
    cache_dir  = get_cache_dir(cache_key)

    vs = FAISSVectorStore()

    if (not rebuild_index) and vs.load(cache_dir):
        status_box.success("  Loaded the search index from cache (faster).")
    else:
        all_chunks = []

        for f in uploaded_files:
            file_bytes = f.getvalue()
            fname      = f.name

            status_box.info(f"📖 Reading PDF text: {fname}")
            try:
                pages = load_pdf_pages(file_bytes, fname)
            except Exception as exc:
                st.warning(f"  Could not extract text from {fname}: {exc}. Skipping.")
                continue

            if not pages:
                st.warning(f"  No text found in {fname} (possibly scanned). Skipping.")
                continue

            status_box.info(f" Chunking: {fname}")
            chunks = chunk_pages(
                pages,
                fname,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            all_chunks.extend(chunks)

        if not all_chunks:
            st.error("  No text could be extracted from any uploaded PDF.")
            st.stop()

        status_box.info(f"  Building search index from {len(all_chunks)} chunks...")
        vs.build(all_chunks)
        vs.save(cache_dir)
        status_box.success(f"  Search index built & cached ({len(all_chunks)} chunks).")

    # ─── Analyze each paper ────────────────────────────────────────────────
    for f in uploaded_files:
        fname = f.name
        st.divider()
        st.subheader(f"  Paper: {fname}")
        paper_log = st.empty()

        # Default-argument trick avoids late-binding closure bug
        def progress_cb(msg: str, _log=paper_log):
            _log.info(msg)

        try:
            analysis = analyze_single_paper(
                vs=vs,
                paper_name=fname,
                provider=provider,
                model=model_name,
                top_k=top_k,
                progress_cb=progress_cb,
            )
            all_analyses.append(analysis)
            paper_log.success(f"  Finished analyzing: {fname}")
        except Exception as exc:
            paper_log.error(f"  Error analyzing {fname}: {exc}")
            continue

        with st.expander(f" Results for: {fname}", expanded=True):
            st.markdown(
                "<div class='mini'>These sections are generated from the paper content.</div>",
                unsafe_allow_html=True,
            )
            col1, col2 = st.columns(2)

            with col1:
                card(" TL;DR (Short summary)",
                     analysis.tldr,
                     help_text="A quick summary in simple terms.")
                st.markdown("<br/>", unsafe_allow_html=True)
                card(" Methods Used (How they did it)",
                     analysis.methods_used,
                     help_text="The approach, model, data, or procedure used in the study.")
                st.markdown("<br/>", unsafe_allow_html=True)
                card("🚧 Limitations (What the study could not do)",
                     analysis.limitations,
                     help_text="Constraints or weaknesses mentioned in the paper.")

            with col2:
                card(" Key Findings (Main results)",
                     analysis.key_findings,
                     help_text="The most important results or conclusions.")
                st.markdown("<br/>", unsafe_allow_html=True)
                card(" Research Gap (What's missing)",
                     analysis.research_gap,
                     help_text="What the paper suggests is not solved yet or needs more work.")
                st.markdown("<br/>", unsafe_allow_html=True)
                card(" Future Work (Next steps)",
                     analysis.future_work,
                     help_text="Ideas the authors suggest for future research.")

    # ─── Multi-paper Literature Review ───────────────────────────────────────
    if len(all_analyses) > 1:
        st.divider()
        st.header("  Literature Review (Across Multiple Papers)")
        st.markdown(
            "<div class='mini'>Compares papers and highlights common themes, differences, and shared gaps.</div>",
            unsafe_allow_html=True,
        )
        lit_log = st.empty()
        lit_log.info("🧩 Creating the literature review...")

        try:
            lit_review = generate_literature_review(
                analyses=all_analyses,
                provider=provider,
                model=model_name,
                progress_cb=lambda m: lit_log.info(m),
            )
            lit_log.success("  Literature review complete.")

            with st.expander("  View Literature Review", expanded=True):
                card("  Overview", lit_review.overview,
                     help_text="High-level summary across all uploaded papers.")
                st.markdown("<br/>", unsafe_allow_html=True)
                card(" Method Comparison", lit_review.method_comparison,
                     help_text="How approaches differ across papers.")
                st.markdown("<br/>", unsafe_allow_html=True)
                card(" Common Findings", lit_review.common_findings,
                     help_text="Findings that appear in multiple papers.")
                st.markdown("<br/>", unsafe_allow_html=True)
                card(" Cross-Paper Research Gaps", lit_review.cross_paper_gaps,
                     help_text="Gaps that remain across studies.")

        except Exception as exc:
            lit_log.error(f"  Error generating literature review: {exc}")

    st.divider()
    st.success(" Done! Your papers have been analyzed.")
