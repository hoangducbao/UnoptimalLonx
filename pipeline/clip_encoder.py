"""
clip_encoder.py — query-time text encoder for the CLIP ViT-B/32 backend
(AICData/clip-features-32/*.npy), via Multilingual-CLIP
(M-CLIP/XLM-Roberta-Large-Vit-B-32). Its text tower is trained to land in
the SAME embedding space as OpenAI's untouched ViT-B/32 image tower, so the
organizer-provided .npy features stay valid with zero re-extraction, and it
understands Vietnamese natively.

Loading is a manual workaround, ported from old_version/pipeline/
text_encoder.py: multilingual-clip's own `.from_pretrained()` call is
incompatible with transformers>=5's meta-device lazy loading (its __init__
does a second, real from_pretrained() call for the underlying
xlm-roberta-large weights, which transformers now rejects while already
inside a meta-device context). Workaround: construct MultilingualCLIP(config)
directly, then load the fine-tuned state dict on top.
"""

import numpy as np

import config

_model = None
_tokenizer = None


def _get_model():
    global _model, _tokenizer
    if _model is None:
        import torch
        import transformers
        from huggingface_hub import hf_hub_download
        from multilingual_clip import Config_MCLIP, pt_multilingual_clip

        mclip_config = Config_MCLIP.MCLIPConfig.from_pretrained(config.CLIP_MCLIP_MODEL_NAME)
        model = pt_multilingual_clip.MultilingualCLIP(mclip_config)

        ckpt_path = hf_hub_download(config.CLIP_MCLIP_MODEL_NAME, filename="pytorch_model.bin")
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # strict=False: the checkpoint carries a "transformer.embeddings.position_ids"
        # key that older transformers versions registered as a persistent buffer
        # (just torch.arange(...), not a learned weight) and newer ones don't --
        # a benign version-skew artifact. Any OTHER mismatch is a real problem.
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        unexpected = [k for k in unexpected if k != "transformer.embeddings.position_ids"]
        if missing or unexpected:
            raise RuntimeError(
                f"Unexpected M-CLIP state_dict mismatch beyond the known position_ids "
                f"quirk -- missing={missing}, unexpected={unexpected}"
            )

        model.eval()
        _model = model
        _tokenizer = transformers.AutoTokenizer.from_pretrained(config.CLIP_MCLIP_MODEL_NAME)
    return _model, _tokenizer


def encode_text(texts: list) -> np.ndarray:
    """(N, 512) float32, L2-normalized -- matches AICData/clip-features-32's
    ViT-B/32 image-embedding space."""
    import torch

    model, tokenizer = _get_model()
    with torch.no_grad():
        embeddings = model.forward(texts, tokenizer)
    vecs = embeddings.cpu().numpy().astype("float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "a man riding a motorbike"
    vecs = encode_text([query])
    print(f"query={query!r} -> shape={vecs.shape} dtype={vecs.dtype} "
          f"norm={np.linalg.norm(vecs[0]):.4f}")
