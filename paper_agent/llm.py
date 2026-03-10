"""
llm.py — LLM client factory supporting Groq, HuggingFace, OpenAI-compatible,
          Gemini, and Anthropic.

Returns a simple callable: chat(user_message: str) -> str
The system prompt from prompts.py is always prepended.
"""

from __future__ import annotations
import logging
import os

from paper_agent.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def make_chat_fn(provider: str, model: str, api_key: str | None, api_base: str | None = None):
    """
    Return a chat(user_message) -> str callable for the given provider.

    Parameters
    ----------
    provider:  one of "groq", "huggingface", "OpenAI-compatible", "Gemini", "Anthropic"
    model:     model name string
    api_key:   provider API key (falls back to env vars if None)
    api_base:  optional custom base URL (for OpenAI-compatible endpoints)
    """

    provider_lower = provider.lower()

    # ── Groq ──────────────────────────────────────────────────────────────────
    if provider_lower == "groq":
        from openai import OpenAI
        key = api_key or os.getenv("GROQ_API_KEY", "")
        client = OpenAI(
            api_key=key,
            base_url="https://api.groq.com/openai/v1",
        )

        def _groq_chat(user_message: str) -> str:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            return resp.choices[0].message.content or ""

        return _groq_chat

    # ── HuggingFace Inference API ─────────────────────────────────────────────
    elif provider_lower == "huggingface":
        key = api_key or os.getenv("HUGGINGFACE_API_KEY", "")

        try:
            # huggingface_hub >= 0.22 has InferenceClient
            from huggingface_hub import InferenceClient
            hf_client = InferenceClient(model=model, token=key)

            def _hf_chat(user_message: str) -> str:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ]
                resp = hf_client.chat_completion(messages=messages, max_tokens=2048)
                return resp.choices[0].message.content or ""

            return _hf_chat

        except (ImportError, AttributeError):
            # Fallback: use openai-compat endpoint for HF Serverless Inference
            from openai import OpenAI
            hf_openai = OpenAI(
                api_key=key,
                base_url="https://api-inference.huggingface.co/v1",
            )

            def _hf_openai_chat(user_message: str) -> str:
                resp = hf_openai.chat.completions.create(
                    model=model,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                )
                return resp.choices[0].message.content or ""

            return _hf_openai_chat

    # ── OpenAI-compatible ─────────────────────────────────────────────────────
    elif provider_lower == "openai-compatible":
        from openai import OpenAI
        key = api_key or os.getenv("OPENAI_API_KEY", "")
        kwargs: dict = {"api_key": key}
        if api_base:
            kwargs["base_url"] = api_base

        client = OpenAI(**kwargs)

        def _openai_chat(user_message: str) -> str:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
            )
            return resp.choices[0].message.content or ""

        return _openai_chat

    # ── Gemini ────────────────────────────────────────────────────────────────
    elif provider_lower == "gemini":
        import google.generativeai as genai
        key = api_key or os.getenv("GOOGLE_API_KEY", "")
        genai.configure(api_key=key)
        gem_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=SYSTEM_PROMPT,
        )

        def _gemini_chat(user_message: str) -> str:
            resp = gem_model.generate_content(user_message)
            return resp.text or ""

        return _gemini_chat

    # ── Anthropic ─────────────────────────────────────────────────────────────
    elif provider_lower == "anthropic":
        import anthropic
        key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        client = anthropic.Anthropic(api_key=key)

        def _anthropic_chat(user_message: str) -> str:
            msg = client.messages.create(
                model=model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            return msg.content[0].text if msg.content else ""

        return _anthropic_chat

    else:
        raise ValueError(f"Unknown provider: {provider!r}")
