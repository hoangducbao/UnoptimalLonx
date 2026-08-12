"""
text_encoder.py — turn a (Vietnamese or English) query string into a 512-d
vector directly comparable to the precomputed CLIP ViT-B/32 image vectors
in clip-features-32/*.npy.

Two methods, same output shape/normalization:

  "mclip" (default) — Multilingual CLIP (M-CLIP/XLM-Roberta-Large-Vit-B-32).
      Its text tower is trained to land in the SAME embedding space as
      OpenAI's untouched ViT-B/32 image tower, so the organizer-provided
      .npy features stay valid with zero re-extraction. Understands
      Vietnamese natively — no network call needed at query time once the
      checkpoint is cached locally (first call downloads ~1-2GB from
      Hugging Face).

  "translate" — the original workaround: deep-translator (Google Translate)
      to English, then OpenAI's own English-centric CLIP text tower. Kept
      as an explicit fallback (e.g. if you're offline and mclip's checkpoint
      isn't cached yet, or as an A/B sanity check against mclip). Requires
      network access at query time for the translation call.

Both paths L2-normalize their output so cosine similarity == inner product,
matching how clip-features-32/*.npy and index_pipeline.py's FAISS vectors
are normalized.
"""

import numpy as np

_mclip_model = None
_mclip_tokenizer = None
_openai_clip_model = None
_openai_clip_device = None

MCLIP_MODEL_NAME = "M-CLIP/XLM-Roberta-Large-Vit-B-32"


def _get_mclip():
    """
    Loads M-CLIP manually instead of via `MultilingualCLIP.from_pretrained(...)`.

    The straightforward call is broken on current transformers (5.x):
    `PreTrainedModel.from_pretrained` wraps model construction in a "meta
    device" context for lazy weight loading, but `MultilingualCLIP.__init__`
    itself does a second, real `AutoModel.from_pretrained(config.modelBase)`
    call (loading the underlying xlm-roberta-large weights) -- and doing a
    real weight-loading from_pretrained call *while already inside* a meta-
    device context is something transformers now explicitly rejects
    ("You are using `from_pretrained` with a meta device context manager").
    multilingual-clip (last updated ~2022) predates this transformers
    behavior, and pinning transformers back to a compatible pre-2024 release
    isn't viable here either -- its `tokenizers` pin has no prebuilt wheel
    for this Python version and fails building from source without a Rust
    toolchain.

    Workaround: construct MultilingualCLIP(config) directly (not via
    .from_pretrained), which runs __init__'s nested from_pretrained call
    with no meta-device context active (so it loads normally), then load
    M-CLIP's own fine-tuned full state dict on top from the (already cached)
    checkpoint. This reuses the ~2.1GB checkpoint downloaded once by
    huggingface_hub -- no re-download.
    """
    global _mclip_model, _mclip_tokenizer
    if _mclip_model is None:
        import torch
        import transformers
        from huggingface_hub import hf_hub_download
        from multilingual_clip import Config_MCLIP, pt_multilingual_clip

        # AutoConfig doesn't know the custom "M-CLIP" model_type (it's
        # registered by the multilingual_clip package, not transformers'
        # own registry) -- load via its own config class directly instead.
        config = Config_MCLIP.MCLIPConfig.from_pretrained(MCLIP_MODEL_NAME)
        model = pt_multilingual_clip.MultilingualCLIP(config)

        ckpt_path = hf_hub_download(MCLIP_MODEL_NAME, filename="pytorch_model.bin")
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # strict=False: the checkpoint carries a "transformer.embeddings.position_ids"
        # key that older transformers versions registered as a persistent buffer
        # (just torch.arange(...), not a learned weight) and newer ones don't --
        # a benign version-skew artifact, safe to ignore. Any OTHER mismatch
        # would be a real problem, so surface it loudly rather than swallowing it.
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        unexpected = [k for k in unexpected if k != "transformer.embeddings.position_ids"]
        if missing or unexpected:
            raise RuntimeError(
                f"Unexpected M-CLIP state_dict mismatch beyond the known position_ids "
                f"quirk -- missing={missing}, unexpected={unexpected}"
            )

        _mclip_model = model
        _mclip_tokenizer = transformers.AutoTokenizer.from_pretrained(MCLIP_MODEL_NAME)
        _mclip_model.eval()
    return _mclip_model, _mclip_tokenizer


def _get_openai_clip():
    global _openai_clip_model, _openai_clip_device
    if _openai_clip_model is None:
        import clip
        import torch

        _openai_clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        _openai_clip_model, _ = clip.load("ViT-B/32", device=_openai_clip_device)
    return _openai_clip_model, _openai_clip_device


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def encode_mclip(texts: list) -> np.ndarray:
    """texts -> (len(texts), 512) float32, L2-normalized. Understands
    Vietnamese directly; no translation step."""
    import torch

    model, tokenizer = _get_mclip()
    with torch.no_grad():
        embeddings = model.forward(texts, tokenizer)
    vecs = embeddings.cpu().numpy().astype("float32")
    return _l2_normalize(vecs)


def encode_translate(texts: list, source: str = "auto") -> np.ndarray:
    """texts -> (len(texts), 512) float32, L2-normalized. Translates each
    string to English (deep-translator/Google) then encodes with OpenAI
    CLIP's ViT-B/32 text tower. Network-dependent."""
    import faiss
    import torch
    from deep_translator import GoogleTranslator

    model, device = _get_openai_clip()
    translator = GoogleTranslator(source=source, target="en")
    translated = [translator.translate(t) for t in texts]

    import clip

    tokens = clip.tokenize(translated, truncate=True).to(device)
    with torch.no_grad():
        embeddings = model.encode_text(tokens)
    vecs = embeddings.cpu().numpy().astype("float32")
    faiss.normalize_L2(vecs)
    return vecs


def encode_text_query(text: str, method: str = "mclip") -> np.ndarray:
    """Single-query convenience wrapper. Returns shape (1, 512) float32,
    L2-normalized, ready for a FAISS IndexFlatIP search."""
    if method == "mclip":
        return encode_mclip([text])
    elif method == "translate":
        return encode_translate([text])
    else:
        raise ValueError(f"unknown method: {method!r} (expected 'mclip' or 'translate')")


if __name__ == "__main__":
    # Quick A/B sanity check: compare methods on the same query this session
    # already validated manually (translate-based results surfaced HTV
    # traffic-news content for this query).
    import argparse

    parser = argparse.ArgumentParser(description="Encode a text query and print vector stats.")
    parser.add_argument("query", type=str, nargs="?", default="cảnh sát giao thông")
    parser.add_argument("--method", choices=["mclip", "translate", "both"], default="both")
    args = parser.parse_args()

    methods = ["mclip", "translate"] if args.method == "both" else [args.method]
    for m in methods:
        vec = encode_text_query(args.query, method=m)
        print(f"[{m}] shape={vec.shape} dtype={vec.dtype} norm={np.linalg.norm(vec):.4f}")
