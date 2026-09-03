"""
backend/models.py -- shared model loaders. Ported from ui/app.py:121,
285-327. @st.cache_resource -> plain module-level singleton (`_siglip2`),
built once by backend/main.py's lifespan hook instead of lazily on first
call -- a FastAPI process has a real one-time startup, so there's no need
for Streamlit's per-process-cache-keyed-by-args dance.
"""

import re

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
    """Encode already-short texts. Anything past the text tower's context
    window is truncated away, so a caller with a possibly-long query should
    go through siglip2_query_mat() instead, which chunks first."""
    model, processor = load_siglip2()
    # max_length is pinned rather than left to the processor's default: every
    # SigLIP2 checkpoint's tokenizer_config.json reports the 1e19 "no limit"
    # sentinel for model_max_length, so the real 64 comes from an undocumented
    # default inside Siglip2Processor. _get_siglip2_max_tokens() measures it.
    inputs = processor(text=texts, padding="max_length", truncation=True,
                       max_length=_get_siglip2_max_tokens(), return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
    feats = out.pooler_output if hasattr(out, "pooler_output") else out
    return feats.float().cpu().numpy().astype("float32")


# ---------------------------------------------------------------------------
# SigLIP2 text-tower token limit
# ---------------------------------------------------------------------------
# The text tower has a hard 64-token context: Siglip2TextModel's
# position_embedding is nn.Embedding(64, hidden), so there is no max_length to
# raise and no RoPE to extrapolate. encode_text_siglip2's truncation=True
# above drops everything past it.
#
# A long query no longer has to lose that text. chunk_text() below splits it
# into sentence-aligned pieces that each fit the window, and
# QUERY_CHUNK_STRATEGIES decides what happens to the pieces -- the same three
# options AICPreprocess/summary-embed.ipynb offers on the corpus side, where
# chunk_text() and _split_long_unit() come from:
#
#   truncate         one vector, first 64 tokens only. The old behaviour,
#                    kept for comparison -- a long query keeps whatever it
#                    happened to say first and silently loses the rest.
#   mean_chunks      one vector: every chunk embedded, L2-normalized and
#                    averaged. A soft conjunction -- a video has to look
#                    somewhat like all of the query, not just part of it.
#                    Cheapest to consume: still one query vector, so every
#                    leg searches exactly as it always did.
#   chunks_separate  N vectors, one per chunk. Each chunk gets its own
#                    ranked list and the lists are fused with RRF
#                    (common.py's faiss_search_pooled) -- the same fusion
#                    the signals already use across their legs. A row is
#                    rewarded for ranking well against several chunks rather
#                    than for one strong match, so the chunks act as
#                    corroborating evidence. DEFAULT.
#
# The notebook max-pools its chunks, and on the corpus side that is right: a
# summary's chunks are unrelated topics, so a hit on one is a real hit. On the
# query side max-pooling would make added clauses behave like an OR -- one
# strongly-matching clause could carry a row that ignored the rest -- so the
# chunk lists are RRF-fused instead, which pays a row for placing well against
# several chunks at once. mean_chunks reaches a similar conjunction in vector
# space rather than in rank space: it is the blunter of the two (four clauses
# average into one point that may sit near none of them) but it keeps a real
# cosine score, where chunks_separate returns an RRF score. Both beat
# truncate, which just deletes the tail.
#
# siglip2_long_query_note() reports when a query was over the window and what
# was done about it -- worth surfacing either way, since the Elasticsearch
# legs see the raw string regardless. See backend/routes/search.py and
# trake.py's `warning` fields.
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


# Sentence boundaries first: a query's sentences are usually one clause of
# the scene each, so sentence-aligned chunks come out as coherent units
# rather than arbitrary 64-token windows.
_SENT_SPLIT = re.compile(r"(?<=[.!?\u2026])\s+|\n+")


def _token_len(text: str, tokenizer) -> int:
    """Length in tokens with truncation OFF, so we can see how far over the
    window a text really is. (The tokenizer prints a one-time "sequence
    longer than model_max_length" warning here; harmless -- nothing measured
    by this function is fed to the model.)"""
    return len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])


def _split_long_unit(unit: str, tokenizer, max_len: int) -> list:
    """Last resort: one sentence is itself over the window -> pack words
    greedily instead. Rare on well-punctuated prose, but a typed query is
    often a single unpunctuated run-on, so this path matters more here than
    it does in the notebook this came from."""
    out, cur = [], ""
    for w in unit.split():
        cand = f"{cur} {w}".strip()
        if cur and _token_len(cand, tokenizer) > max_len:
            out.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        out.append(cur)
    return out


def chunk_text(text: str, tokenizer=None, max_len: int = None) -> list:
    """Greedily pack sentences into chunks that each fit `max_len` tokens.
    Returns [text] unchanged when it already fits, so the common short-query
    case costs one tokenizer call and nothing else."""
    if tokenizer is None:
        _, processor = load_siglip2()
        tokenizer = processor.tokenizer
    max_len = max_len or _get_siglip2_max_tokens()
    if _token_len(text, tokenizer) <= max_len:
        return [text]

    units = [u.strip() for u in _SENT_SPLIT.split(text) if u and u.strip()] or [text]
    chunks, cur = [], ""
    for unit in units:
        cand = f"{cur} {unit}".strip()
        if _token_len(cand, tokenizer) <= max_len:
            cur = cand
            continue
        if cur:
            chunks.append(cur)
            cur = ""
        if _token_len(unit, tokenizer) <= max_len:
            cur = unit
        else:
            chunks.extend(_split_long_unit(unit, tokenizer, max_len))
    if cur:
        chunks.append(cur)
    return chunks or [text]


QUERY_CHUNK_STRATEGIES = ("truncate", "mean_chunks", "chunks_separate")
DEFAULT_QUERY_CHUNK_STRATEGY = "chunks_separate"

# Process-global rather than per-request, deliberately: the profile
# (config.EMBED_PROFILE) already works this way, and like the profile this is
# a property of how the process searches, not of one query. The cost is that
# two browser tabs pointed at the same port share it -- the settings dialog
# reads the live value back on open so it can't silently disagree.
_query_chunk_strategy = DEFAULT_QUERY_CHUNK_STRATEGY


def get_query_chunk_strategy() -> str:
    return _query_chunk_strategy


def set_query_chunk_strategy(name: str) -> str:
    """Switch the active strategy. Every leg's result cache keys on it
    (common.py's query_hash), so a switch is visible on the very next search
    instead of being masked by a cached ranking from the old one."""
    global _query_chunk_strategy
    if name not in QUERY_CHUNK_STRATEGIES:
        raise ValueError(f"unknown query chunk strategy {name!r} -- "
                         f"expected one of {list(QUERY_CHUNK_STRATEGIES)}")
    _query_chunk_strategy = name
    return _query_chunk_strategy


def siglip2_long_query_tokens(text: str):
    """Token count of `text` if it overflows SigLIP2's text-tower window,
    None if it fits. Callers that only need "is this over the window, and by
    how much" use this; siglip2_long_query_note() wraps it into a sentence."""
    if not text or not text.strip():
        return None
    _, processor = load_siglip2()
    max_tokens = _get_siglip2_max_tokens()
    token_count = _token_len(text, processor.tokenizer)
    return token_count if token_count > max_tokens else None


def siglip2_long_query_note(text: str):
    """None if `text` fits SigLIP2's text-tower context window in one piece;
    otherwise a short user-facing note giving the query's token count and the
    chunking strategy handling the overflow. Text-only: image queries go
    through encode_image_siglip2, which has no token limit, so callers skip
    this for an is_image_query() query."""
    token_count = siglip2_long_query_tokens(text)
    if token_count is None:
        return None
    return (f"Query is {token_count} tokens, over SigLIP2's "
            f"{_get_siglip2_max_tokens()}-token window -- '{get_query_chunk_strategy()}'.")


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


def _l2(x, axis=-1, eps=1e-12):
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)


def siglip2_query_mat(query) -> np.ndarray:
    """The (n_vectors, dim) query matrix every SigLIP2-backed leg searches
    with -- hand it to common.py's faiss_search_pooled().

    Three branch points live here so no leg has to know about any of them:
      * an uploaded/pasted image goes through the image tower instead of the
        text tower (same joint space, so the result is interchangeable);
      * a text query that fits the 64-token window is encoded once, producing
        a vector bit-for-bit identical to what this returned before chunking
        existed -- short-query rankings are unchanged under every strategy;
      * a longer one is chunked and resolved per the active strategy (see the
        block comment above), giving one row for truncate/mean_chunks and one
        row per chunk for chunks_separate.
    """
    if is_image_query(query):
        return encode_image_siglip2([query])
    if get_query_chunk_strategy() == "truncate":
        return encode_text_siglip2([query])
    _, processor = load_siglip2()
    chunks = chunk_text(query, processor.tokenizer)
    if len(chunks) == 1:
        return encode_text_siglip2([query])
    embeds = _l2(encode_text_siglip2(chunks))
    if get_query_chunk_strategy() == "chunks_separate":
        return embeds
    # mean_chunks: already normalized per chunk above so a chunk with a larger
    # raw norm can't dominate the mean; renormalize after averaging.
    return _l2(embeds.mean(axis=0)).reshape(1, -1)


def siglip2_query_vec(query) -> np.ndarray:
    """Single-vector form of siglip2_query_mat(), for a caller that can only
    hold one query vector. Under chunks_separate that means collapsing the
    chunks to their mean, i.e. quietly falling back to mean_chunks -- prefer
    siglip2_query_mat() wherever the search can take a matrix."""
    mat = siglip2_query_mat(query)
    return mat[0] if mat.shape[0] == 1 else _l2(mat.mean(axis=0))
