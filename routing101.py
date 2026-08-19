"""
routing101.py — minimal, single-embedding-space text -> frame search.

Deliberately separate from the weighted-RRF pipeline: does NOT import
pipeline/fusion.py or pipeline/retrieve.py. No RRF, no object-class leg,
no FTS5 text leg — one backend, one FAISS IndexFlatIP, raw cosine ranking.

It DOES reuse pipeline/config.py, pipeline/loader.py, pipeline/
viclip_encoder.py and pipeline/clip_encoder.py: those are plain
paths/text-tower modules with no fusion logic in them, so importing them
doesn't pull any RRF/object-class/FTS5 behavior along with them.

Three backends, selected one at a time with --backend:
  siglip2       google/siglip2-base-patch16-384 image + text towers.
                No prebuilt frame index exists for this repo yet, so the
                frame FAISS index is built on first use by encoding
                AICData/keyframes/{video_id}/{filename} images directly.
  clip_vitb32   AICData/clip-features-32/*.npy (precomputed) + Multilingual
                -CLIP text tower (pipeline/clip_encoder.py).
  viclip        AICDataExtracted/embeddings/*_viclip768.npy (precomputed) +
                ViCLIP-OT text tower (pipeline/viclip_encoder.py).

Each backend gets its own FAISS index + metadata sidecar CSV under
index/routing101/ (built once, cached on disk, reused after) — kept
separate from pipeline/index_pipeline.py's shared SQLite store so this
script's global_id numbering and results can't be contaminated by, or
contaminate, the object-class/FTS5 legs.

CLI:
    python routing101.py "a man riding a motorbike" --backend siglip2 --k 100
    python routing101.py --eval queries.csv --backend clip_vitb32
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("routing101")

REPO_ROOT = Path(__file__).resolve().parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))  # so pipeline/*.py's bare `import config` etc. resolve

import config  # noqa: E402  (pipeline/config.py)
import loader  # noqa: E402  (pipeline/loader.py)

INDEX_DIR = REPO_ROOT / "index" / "routing101"
SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-384"
IMAGE_BATCH_SIZE = 32

BACKENDS = ("siglip2", "clip_vitb32", "viclip")


# ---------------------------------------------------------------------------
# Text encoders (one text tower per backend, all query-time only)
# ---------------------------------------------------------------------------

def _encode_text_clip_vitb32(texts: list) -> np.ndarray:
    import clip_encoder
    return clip_encoder.encode_text(texts)


def _encode_text_viclip(texts: list) -> np.ndarray:
    import viclip_encoder
    return viclip_encoder.encode_text(texts)


_siglip2_state = {}  # lazy singleton: {"model", "processor", "device"}


def _get_siglip2():
    if not _siglip2_state:
        import torch
        from transformers import AutoModel, AutoProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModel.from_pretrained(SIGLIP2_MODEL_ID)
        model.to(device)
        model.eval()
        processor = AutoProcessor.from_pretrained(SIGLIP2_MODEL_ID)
        _siglip2_state.update(model=model, processor=processor, device=device)
    return _siglip2_state["model"], _siglip2_state["processor"], _siglip2_state["device"]


def _encode_text_siglip2(texts: list) -> np.ndarray:
    import torch

    model, processor, device = _get_siglip2()
    inputs = processor(
        text=texts, padding="max_length", truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
    # transformers' SiglipModel.get_text_features returns the raw
    # BaseModelOutputWithPooling (delegates straight to the text tower)
    # rather than a plain tensor -- pooler_output is the pooled embedding.
    feats = out.pooler_output if hasattr(out, "pooler_output") else out
    return feats.float().cpu().numpy().astype("float32")


def _encode_images_siglip2(images: list) -> np.ndarray:
    import torch

    model, processor, device = _get_siglip2()
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.get_image_features(**inputs)
    feats = out.pooler_output if hasattr(out, "pooler_output") else out
    return feats.float().cpu().numpy().astype("float32")


TEXT_ENCODERS = {
    "siglip2": _encode_text_siglip2,
    "clip_vitb32": _encode_text_clip_vitb32,
    "viclip": _encode_text_viclip,
}


# ---------------------------------------------------------------------------
# Frame index build / load — each backend gets its own FAISS file + CSV
# sidecar (global_id, video_id, filename, n, frame_idx, pts_time) under
# index/routing101/, independent of pipeline/index_pipeline.py's DB.
# ---------------------------------------------------------------------------

def _paths(backend: str):
    return (
        INDEX_DIR / f"frame_{backend}_flat_ip.index",
        INDEX_DIR / f"meta_{backend}.csv",
    )


def _write_meta(meta_path: Path, rows: list) -> None:
    df = pd.DataFrame(rows, columns=["global_id", "video_id", "filename", "n", "frame_idx", "pts_time"])
    df.to_csv(meta_path, index=False)


def _build_precomputed_index(backend: str, faiss_path: Path, meta_path: Path) -> None:
    """clip_vitb32 / viclip: vectors are already on disk (pipeline/loader.py);
    just re-embed them into this script's own FAISS file + metadata sidecar."""
    import faiss

    load_fn = loader.load_video_clip_vitb32 if backend == "clip_vitb32" else loader.load_video_viclip
    dim = config.BACKENDS[backend]["dim"]
    video_ids = loader.discover_video_ids(config.EMBEDDINGS_DIR)
    logger.info("[%s] building frame index over %d candidate videos", backend, len(video_ids))

    index = faiss.IndexFlatIP(dim)
    rows = []
    gid = 0
    t0 = time.monotonic()
    for i, video_id in enumerate(video_ids):
        try:
            result = load_fn(video_id)
        except (FileNotFoundError, AssertionError) as e:
            logger.warning("skipping %s: %s", video_id, e)
            continue
        if not result.records:
            continue
        vecs = np.vstack([r.embedding for r in result.records]).astype("float32")
        import faiss as _faiss
        _faiss.normalize_L2(vecs)
        index.add(vecs)
        for r in result.records:
            rows.append((gid, video_id, r.filename, r.n, r.frame_idx, r.pts_time))
            gid += 1
        if (i + 1) % 100 == 0:
            logger.info("[%s] %d/%d videos, %d frames indexed (%.1fs)",
                        backend, i + 1, len(video_ids), gid, time.monotonic() - t0)

    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_path))
    _write_meta(meta_path, rows)
    logger.info("[%s] done: %d frames -> %s", backend, gid, faiss_path)


def _build_siglip2_index(faiss_path: Path, meta_path: Path) -> None:
    """No precomputed siglip2 frame vectors exist -- encode keyframe images
    directly with the SigLIP2 image tower. Frames whose thumbnail file is
    missing are skipped (can't be encoded) rather than aborting the build."""
    import faiss
    from PIL import Image

    video_ids = loader.discover_video_ids(config.EMBEDDINGS_DIR)
    logger.info("[siglip2] building frame index over %d candidate videos "
                "(encoding images with %s -- this is slow)", len(video_ids), SIGLIP2_MODEL_ID)

    index = None
    rows = []
    gid = 0
    t0 = time.monotonic()
    batch_images, batch_meta = [], []

    def _flush():
        nonlocal index, gid
        if not batch_images:
            return
        feats = _encode_images_siglip2(batch_images)
        faiss.normalize_L2(feats)
        if index is None:
            index = faiss.IndexFlatIP(feats.shape[1])
        index.add(feats)
        for (video_id, filename, n, frame_idx, pts_time) in batch_meta:
            rows.append((gid, video_id, filename, n, frame_idx, pts_time))
            gid += 1
        batch_images.clear()
        batch_meta.clear()

    for i, video_id in enumerate(video_ids):
        try:
            # loader.load_video_viclip is used only for its filename/timestamp
            # listing (real filenames + map-keyframes join) -- its embedding
            # vectors are ignored here.
            result = loader.load_video_viclip(video_id)
        except (FileNotFoundError, AssertionError) as e:
            logger.warning("skipping %s: %s", video_id, e)
            continue
        for r in result.records:
            img_path = config.THUMBNAIL_ROOT / video_id / r.filename
            if not img_path.exists():
                continue  # can't encode a missing image
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.warning("skipping %s/%s: %s", video_id, r.filename, e)
                continue
            batch_images.append(img)
            batch_meta.append((video_id, r.filename, r.n, r.frame_idx, r.pts_time))
            if len(batch_images) >= IMAGE_BATCH_SIZE:
                _flush()
        if (i + 1) % 20 == 0:
            logger.info("[siglip2] %d/%d videos, %d frames indexed (%.1fs)",
                        i + 1, len(video_ids), gid, time.monotonic() - t0)
    _flush()

    if index is None:
        raise RuntimeError("siglip2: no frames could be encoded -- check AICData/keyframes")

    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_path))
    _write_meta(meta_path, rows)
    logger.info("[siglip2] done: %d frames -> %s", gid, faiss_path)


_loaded = {}  # backend -> (faiss_index, meta_df)


def load_backend_index(backend: str):
    """Load this backend's FAISS index + metadata, building it first if
    missing. Cached in-process so --eval doesn't rebuild/reload per query."""
    if backend in _loaded:
        return _loaded[backend]

    faiss_path, meta_path = _paths(backend)
    if not faiss_path.exists() or not meta_path.exists():
        if backend == "siglip2":
            _build_siglip2_index(faiss_path, meta_path)
        elif backend in ("clip_vitb32", "viclip"):
            _build_precomputed_index(backend, faiss_path, meta_path)
        else:
            raise ValueError(f"unknown backend {backend!r}; choices: {BACKENDS}")

    import faiss
    index = faiss.read_index(str(faiss_path))
    meta = pd.read_csv(meta_path)
    _loaded[backend] = (index, meta)
    return index, meta


# ---------------------------------------------------------------------------
# Search — single embedding space, raw FAISS IndexFlatIP, no fusion
# ---------------------------------------------------------------------------

def search(query: str, backend: str, k: int = 100) -> list:
    """Returns up to k results, best first:
    [{"global_id", "video_id", "frame_timestamp", "score", "rank"}, ...]
    rank is 1-indexed."""
    import faiss

    index, meta = load_backend_index(backend)
    query_vec = TEXT_ENCODERS[backend]([query])
    faiss.normalize_L2(query_vec)

    n = min(k, index.ntotal)
    scores, ids = index.search(query_vec, n)

    results = []
    for rank, (gid, score) in enumerate(zip(ids[0], scores[0]), start=1):
        if gid == -1:
            continue
        row = meta.iloc[int(gid)]
        results.append({
            "global_id": int(gid),
            "video_id": row["video_id"],
            "frame_timestamp": None if pd.isna(row["pts_time"]) else float(row["pts_time"]),
            "score": float(score),
            "rank": rank,
        })
    return results


# ---------------------------------------------------------------------------
# Evaluation — hit rate @1/@5/@10/@100 + mean rank of the ground-truth frame
# ---------------------------------------------------------------------------

def _frame_matches(meta_row: pd.Series, ground_truth_frame_id) -> bool:
    """Ground-truth frame id is matched against whichever frame identifier
    it looks like: the map-keyframes 1-indexed `n`, the competition
    `frame_idx`, or the keyframe filename's numeric stem -- whichever is
    populated and comparable."""
    gt = str(ground_truth_frame_id).strip()
    if not gt:
        return False
    candidates = set()
    for key in ("n", "frame_idx"):
        val = meta_row[key]
        if pd.notna(val):
            candidates.add(str(int(val)))
    filename = str(meta_row["filename"])
    stem = filename.rsplit(".", 1)[0]
    candidates.add(stem)
    candidates.add(stem.lstrip("0") or "0")
    if gt in candidates:
        return True
    if gt.isdigit():
        gt_int = int(gt)
        for c in candidates:
            if c.isdigit() and int(c) == gt_int:
                return True
    return False


def evaluate(queries_csv: Path, backend: str, k: int = 100) -> dict:
    df = pd.read_csv(queries_csv)
    required = {"query", "ground_truth_video_id", "ground_truth_frame_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{queries_csv} missing required column(s): {sorted(missing)}")

    thresholds = (1, 5, 10, 100)
    hits = {t: 0 for t in thresholds}
    found_ranks = []
    n_queries = len(df)

    for _, row in df.iterrows():
        results = search(str(row["query"]), backend=backend, k=max(k, max(thresholds)))
        gt_video = str(row["ground_truth_video_id"]).strip()
        found_rank = None
        for r in results:
            if str(r["video_id"]).strip() != gt_video:
                continue
            meta_row = _loaded[backend][1].iloc[r["global_id"]]
            if _frame_matches(meta_row, row["ground_truth_frame_id"]):
                found_rank = r["rank"]
                break
        if found_rank is not None:
            found_ranks.append(found_rank)
            for t in thresholds:
                if found_rank <= t:
                    hits[t] += 1

    report = {
        "backend": backend,
        "n_queries": n_queries,
        "n_found": len(found_ranks),
        "hit_rate": {f"@{t}": (hits[t] / n_queries if n_queries else 0.0) for t in thresholds},
        "mean_rank_when_found": (sum(found_ranks) / len(found_ranks)) if found_ranks else None,
    }
    return report


def _print_report(report: dict) -> None:
    print(f"backend={report['backend']}  queries={report['n_queries']}  "
          f"found={report['n_found']}/{report['n_queries']}")
    for label, rate in report["hit_rate"].items():
        print(f"  hit_rate{label:>5s} = {rate:.4f}")
    mr = report["mean_rank_when_found"]
    print(f"  mean_rank_when_found = {mr:.2f}" if mr is not None else "  mean_rank_when_found = n/a")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Minimal single-embedding-space text->frame search (no RRF, no "
                     "object-class leg, no FTS5 text leg). One backend at a time."
    )
    parser.add_argument("query", nargs="?", help="Text query (ignored with --eval).")
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--k", type=int, default=100, help="Top-K results (default 100).")
    parser.add_argument("--eval", type=Path, default=None,
                         help="CSV of (query, ground_truth_video_id, ground_truth_frame_id); "
                              "reports hit rate @1/@5/@10/@100 + mean rank when found.")
    args = parser.parse_args()

    if args.eval is not None:
        report = evaluate(args.eval, backend=args.backend, k=args.k)
        _print_report(report)
    else:
        if not args.query:
            parser.error("query is required unless --eval is given")
        for r in search(args.query, backend=args.backend, k=args.k):
            ts = f"{r['frame_timestamp']:.2f}s" if r["frame_timestamp"] is not None else "?"
            print(f"{r['rank']:3d}  {r['score']:.4f}  {r['video_id']}  @ {ts}  global_id={r['global_id']}")
