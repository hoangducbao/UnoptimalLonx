"""submission/pipeline.py — in-process wrappers over the FastAPI backend's
search modules.

Calling the backend functions directly (instead of an HTTP round-trip) keeps
the whole batch inside ONE process, so the model weights loaded by
backend/models.load_siglip2() plus the multilingual-CLIP text tower exist
once — the same single-process constraint backend/main.py's lifespan enforces
(README/CLAUDE: never duplicate model weights across processes).

warmup() mirrors backend/main.py's lifespan: build the FAISS frame/ASR/
caption/summary indices and ensure the Elasticsearch fuzzy indexes exist. The
search legs already degrade gracefully (empty result + warning) when ES or an
index is unavailable, so a missing asset warns and moves on rather than
killing the whole batch.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO / "pipeline") not in sys.path:
    sys.path.insert(0, str(_REPO / "pipeline"))

# Importing `backend` runs its __init__, which puts pipeline/ on sys.path so
# `import clip_encoder` resolves inside backend.search.keyframe.
from backend import config as bk_config
from backend.search import mixed as mixed_mod
from backend.search import trake as trake_mod


def _warn(msg: str) -> None:
    print(f"[pipeline] warning: {msg}", file=sys.stderr)


def warmup(cfg) -> None:
    """Build/load every index + model the Mixed/TRAKE legs touch. Mirrors
    backend/main.py's lifespan warmup, but never raises — a missing asset just
    marks that leg unusable (each search count returns empty and moves on)."""
    from backend.models import DEVICE

    bk_config.tune_thread_pools(DEVICE)

    # SigLIP2 text/image tower (needed by every SigLIP2-leg search).
    try:
        from backend.models import load_siglip2

        load_siglip2()
    except Exception as e:  # noqa: BLE001
        _warn(f"SigLIP2 model load failed: {e} — SigLIP2-backed legs will be empty.")

    # FAISS indexes (frame, ASR, caption, summary) + ES fuzzy indexes.
    from backend.search import asr as asr_mod
    from backend.search import caption as cap_mod
    from backend.search import keyframe as kf
    from backend.search import summary as sum_mod

    builds = [
        ("Keyframe SigLIP2 frame index", lambda: kf.build_frame_index(bk_config.FRAME_SIGLIP2_GLOB)),
        ("Keyframe CLIP frame index", lambda: kf.build_frame_index(bk_config.FRAME_CLIP_GLOB)),
        ("ASR SigLIP2 index", lambda: asr_mod.build_siglip_asr_index()),
        ("Caption SigLIP2 index", lambda: cap_mod.build_siglip_caption_index()),
        ("Summary embedding + SigLIP2 index", lambda: sum_mod.build_siglip_summary_index()),
    ]
    for label, fn in builds:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            _warn(f"{label} unavailable ({e}) — those legs will be empty.")

    try:
        from backend.es_indexing import ensure_all_fuzzy_indices

        ensure_all_fuzzy_indices()
    except Exception as e:  # noqa: BLE001
        _warn(f"Elasticsearch fuzzy indexes unavailable ({e}); fuzzy legs will be empty.")
# ---------------------------------------------------------------------------
# Mixed-mode search (Keyframe/ASR/Caption/OCR weighted RRF)
# ---------------------------------------------------------------------------

def mixed_results(query: str, cfg, top_n=None):
    """Search `query` with Mixed mode and return ranked rows.

    Returns a list of {"video_id", "n", "rank"} (n = the keyframe number),
    capped at cfg.max_rows. Mirrors backend/routes/search.py::search_mixed
    but in-process, and skips a signal if all its legs come back empty.
    """
    top_n = top_n or cfg.max_rows
    fetch_k = max(cfg.top_k, bk_config.FETCH_K)
    signal_dfs: dict = {}
    w, legs = cfg.weights, cfg.legs

    def _go(fn):
        try:
            df = fn()
            return df if (df is not None and not df.empty) else None
        except Exception as e:  # noqa: BLE001
            _warn(f"signal leg failed: {e}")
            return None

    if w.get("Keyframe", 0):
        df = _go(lambda: mixed_mod._mixed_keyframe_df(query, fetch_k, "", None, legs))
        if df is not None:
            signal_dfs["Keyframe"] = df
    if w.get("ASR", 0):
        df = _go(lambda: mixed_mod._mixed_asr_df(query, fetch_k, "", None, legs))
        if df is not None:
            signal_dfs["ASR"] = df
    if w.get("Caption", 0):
        df = _go(lambda: mixed_mod._mixed_caption_df(query, fetch_k, "", None, legs))
        if df is not None:
            signal_dfs["Caption"] = df
    if w.get("OCR", 0):
        df = _go(lambda: mixed_mod._mixed_ocr_df(query, fetch_k, "", None))
        if df is not None:
            signal_dfs["OCR"] = df

    if not signal_dfs:
        return []

    fused = mixed_mod.rrf_fuse_weighted(signal_dfs, w, top_n=top_n)
    return [
        {"video_id": r["video_id"], "n": int(r["n"]), "rank": int(r["rank"])}
        for _, r in fused.iterrows()
    ]


# ---------------------------------------------------------------------------
# TRAKE — ordered-event video retrieval
# ---------------------------------------------------------------------------

def trake_results(events: list, cfg) -> list:
    """Return rows whose frame ids form a full, chronologically-ordered chain.

    Each element of `events` is one event sub-query; a CodaBench submission row
    needs exactly len(events) frame ids in event order, so candidates that fail
    to match every event are dropped (trake_rank_videos already enforces the
    strict-increasing-time rule between matched events).
    """
    fetch_k = max(cfg.top_k, bk_config.FETCH_K)
    n_events = len(events)
    dfs, labels = [], []
    for i, ev in enumerate(events):
        df = trake_mod.trake_search_event(
            ev, cfg.trake_event_signal, fetch_k, "", None,
            mixed_weights=cfg.weights, mixed_legs=cfg.legs,
        )
        dfs.append(df)
        labels.append(f"E{i + 1}")

    candidates = trake_mod.trake_rank_videos(dfs, labels, cfg.trake_top_videos)

    rows = []
    for c in candidates:
        matched = [e for e in c["events"] if e["matched"]]
        if len(matched) != n_events:
            continue
        rows.append({
            "video_id": c["video_id"],
            "frame_ids": [int(e["n"]) for e in matched],
        })
    return rows