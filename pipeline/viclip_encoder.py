"""
viclip_encoder.py — query-time text encoder, ViCLIP-OT text tower.

Only text encoding is needed at runtime: frame embeddings are precomputed
(AICDataExtracted/embeddings/*_viclip768.npy) and the object-class index is
built once at indexing time (also via this module's encode_text). Nothing
here touches the image tower.

Model: minhnguyent546/ViCLIP-OT (arXiv:2602.22678) — loaded via
`transformers.AutoModel` with `trust_remote_code=True`, custom
`encode_text(texts, normalize=True) -> (N, 768)` method.
"""

import numpy as np
import torch

import config

_model = None
_device = None


def _get_model():
    """Lazy singleton — avoid paying model-load cost (and, on first run, the
    HF Hub download) unless a query is actually being encoded."""
    global _model, _device
    if _model is None:
        from transformers import AutoModel

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = AutoModel.from_pretrained(config.MODEL_ID, trust_remote_code=True)
        _model.to(_device)
        _model.eval()
    return _model


def encode_text(texts: list) -> np.ndarray:
    """Encode a list of query strings into (N, 768) float32, L2-normalized,
    in the same embedding space as the precomputed frame vectors."""
    model = _get_model()
    with torch.no_grad():
        embeddings = model.encode_text(texts, normalize=True)
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    return np.asarray(embeddings, dtype="float32")


if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "một người đàn ông đang lái xe"
    vecs = encode_text([query])
    print(f"query={query!r} -> shape={vecs.shape} dtype={vecs.dtype} "
          f"norm={np.linalg.norm(vecs[0]):.4f}")
