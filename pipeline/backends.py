"""
backends.py — wires config.BACKENDS' paths/dims to the actual per-backend
encode_text/load_video functions. Everything else in the pipeline (index
building, fusion, the UI) is backend-agnostic: it takes a backend key,
looks up what it needs here, and never hardcodes model-specific behavior.
"""

import clip_encoder
import config
import loader
import viclip_encoder

REGISTRY = {
    "viclip": {
        **config.BACKENDS["viclip"],
        "encode_text": viclip_encoder.encode_text,
        "load_video": loader.load_video_viclip,
    },
    "clip_vitb32": {
        **config.BACKENDS["clip_vitb32"],
        "encode_text": clip_encoder.encode_text,
        "load_video": loader.load_video_clip_vitb32,
    },
}


def get(backend: str) -> dict:
    if backend not in REGISTRY:
        raise ValueError(f"unknown backend {backend!r}; choices: {list(REGISTRY)}")
    return REGISTRY[backend]
