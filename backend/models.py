"""
backend/models.py -- shared model loaders. Ported from ui/app.py:121,
285-327. @st.cache_resource -> plain module-level singleton (`_siglip2`),
built once by backend/main.py's lifespan hook instead of lazily on first
call -- a FastAPI process has a real one-time startup, so there's no need
for Streamlit's per-process-cache-keyed-by-args dance.
"""

import numpy as np
import torch
from PIL import Image

from . import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_siglip2 = None  # (model, processor), set by load_siglip2()


def load_siglip2():
    global _siglip2
    if _siglip2 is None:
        from transformers import AutoModel, AutoProcessor

        model = AutoModel.from_pretrained(config.SIGLIP2_MODEL_ID).to(DEVICE).eval()
        processor = AutoProcessor.from_pretrained(config.SIGLIP2_MODEL_ID)
        _siglip2 = (model, processor)
    return _siglip2


def encode_text_siglip2(texts: list) -> np.ndarray:
    model, processor = load_siglip2()
    inputs = processor(text=texts, padding="max_length", truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
    feats = out.pooler_output if hasattr(out, "pooler_output") else out
    return feats.float().cpu().numpy().astype("float32")


# ---------------------------------------------------------------------------
# SigLIP2 text-tower token limit -- encode_text_siglip2's truncation=True
# above silently drops anything past the tower's fixed context window, with
# no signal to the caller that it happened. siglip2_truncation_warning()
# lets a route surface that instead of eating it -- see backend/routes/
# search.py and trake.py's `warning` fields.
# ---------------------------------------------------------------------------

_siglip2_max_tokens = None  # measured once, see _get_siglip2_max_tokens()


def _get_siglip2_max_tokens() -> int:
    """The token count encode_text_siglip2's padding="max_length" actually
    pads/truncates to. Not reliably readable off the tokenizer directly --
    this checkpoint's tokenizer_config.json doesn't set model_max_length, so
    HF's tokenizer reports a sentinel "no limit" value there instead of the
    real one. Measured by forcing truncation on a deliberately oversized
    probe string and reading back the resulting length; cached since it
    can't change at runtime."""
    global _siglip2_max_tokens
    if _siglip2_max_tokens is None:
        _, processor = load_siglip2()
        probe = processor(text=["word " * 500], padding="max_length", truncation=True, return_tensors="pt")
        _siglip2_max_tokens = probe["input_ids"].shape[-1]
    return _siglip2_max_tokens


def siglip2_truncation_warning(text: str):
    """None if `text` fits SigLIP2's text-tower context window untouched;
    otherwise a user-facing warning string. Text-only -- image queries go
    through encode_image_siglip2 instead, no token limit there, so callers
    should skip this for an is_image_query() query."""
    if not text or not text.strip():
        return None
    _, processor = load_siglip2()
    token_count = len(processor.tokenizer(text, truncation=False)["input_ids"])
    max_tokens = _get_siglip2_max_tokens()
    if token_count <= max_tokens:
        return None
    return (
        f"Query is {token_count} tokens -- SigLIP2 embedding legs only use the first "
        f"{max_tokens} (the rest is silently dropped for those legs; fuzzy text legs still "
        f"see the full query)."
    )


def encode_image_siglip2(images: list) -> np.ndarray:
    """SigLIP2 image tower -- same joint text/image embedding space as
    encode_text_siglip2(), so an image query is directly comparable to
    every SigLIP2-embedded leg (frame, ASR, caption, summary), not just
    the frame index."""
    model, processor = load_siglip2()
    inputs = processor(images=images, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.get_image_features(**inputs)
    feats = out.pooler_output if hasattr(out, "pooler_output") else out
    return feats.float().cpu().numpy().astype("float32")


def is_image_query(query) -> bool:
    return isinstance(query, Image.Image)


def siglip2_query_vec(query) -> np.ndarray:
    """Picture-query support: an uploaded/pasted image is embedded with the
    SigLIP2 image tower instead of the text tower -- everywhere else the
    caller already treats the result as a plain query vector, so this is
    the only branch point needed for every SigLIP2-backed leg."""
    if is_image_query(query):
        return encode_image_siglip2([query])[0]
    return encode_text_siglip2([query])[0]
