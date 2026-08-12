"""
retrieve.py — query-time pipeline (spec's 5 steps), backend-selectable
(config.BACKENDS / backends.py):

  1. encode query with the chosen backend's text tower, search that
     backend's frame embedding index
  2. same query vector, search that backend's object-class embedding
     index, map matched classes back to frames via the frame_classes
     join table (shared across backends)
  3. search OCR/caption text via FTS5 (trigram, diacritic-folded; shared)
  4. fuse the (up to three) ranked lists with RRF (fusion.py)
  5. return ranked frame results with video/timestamp/thumbnail info

Each leg degrades gracefully: a leg with no data (empty/missing index,
empty FTS5 table) is simply omitted from fusion rather than raising.

Every leg also accepts an optional video_id, restricting candidates to one
video (the UI's "search in this video only") -- global_id ranges are
contiguous per video (assigned in indexing order), so this is just a range
filter, not a rebuilt index.
"""

import argparse
from pathlib import Path

import faiss
import numpy as np

import backends as backend_registry
import config
import fusion
import store


def load_backend(
    backend: str = config.DEFAULT_BACKEND,
    db_path: Path = config.DB_PATH,
):
    """Open the given backend's frame/class FAISS indices (None if not yet
    built -- that leg then degrades gracefully) and the (backend-agnostic)
    metadata DB."""
    defn = backend_registry.get(backend)
    conn = store.connect(db_path)
    frame_faiss_path = Path(defn["frame_faiss_path"])
    class_faiss_path = Path(defn["class_faiss_path"])
    frame_index = faiss.read_index(str(frame_faiss_path)) if frame_faiss_path.exists() else None
    class_index = faiss.read_index(str(class_faiss_path)) if class_faiss_path.exists() else None
    return frame_index, class_index, conn


def _frame_leg(query_vec, frame_index, top_k: int, id_range=None) -> list:
    if frame_index is None or frame_index.ntotal == 0:
        return []
    if id_range is None:
        _scores, ids = frame_index.search(query_vec, min(top_k, frame_index.ntotal))
        return [int(i) for i in ids[0] if i != -1]
    start, count = id_range
    if count <= 0:
        return []
    vecs = frame_index.reconstruct_n(start, count)  # cheap: one video's rows, IndexFlat only
    sims = vecs @ query_vec[0]
    order = np.argsort(-sims)[: min(top_k, count)]
    return [start + int(i) for i in order]


def _class_leg(query_vec, class_index, conn, top_classes: int, top_k: int, id_range=None) -> list:
    """Search the class index for the best-matching object labels, then rank
    frames containing any of those labels by their best matching class's
    cosine score (a frame matching multiple returned classes takes its max)."""
    if class_index is None or class_index.ntotal == 0:
        return []
    k = min(top_classes, class_index.ntotal)
    scores, ids = class_index.search(query_vec, k)
    class_names = store.get_classes_ordered(conn)
    matched = [(class_names[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
    if not matched:
        return []
    class_score = dict(matched)

    best_frame_score: dict = {}
    for row in store.frames_for_classes(conn, [name for name, _ in matched]):
        match_score = class_score.get(row["class_name"])
        if match_score is None:
            continue
        gid = row["global_id"]
        if id_range is not None:
            start, count = id_range
            if not (start <= gid < start + count):
                continue
        if gid not in best_frame_score or match_score > best_frame_score[gid]:
            best_frame_score[gid] = match_score

    ranked = sorted(best_frame_score.items(), key=lambda kv: kv[1], reverse=True)
    return [gid for gid, _ in ranked[:top_k]]


def _text_leg(query: str, conn, top_k: int, video_id: str | None = None) -> list:
    if conn.execute("SELECT COUNT(*) AS n FROM keyframe_text").fetchone()["n"] == 0:
        return []  # no OCR/caption data yet -- skip rather than run an empty-table query
    return [row["global_id"] for row in store.search_text(conn, query, video_id=video_id, limit=top_k)]


def search(
    query: str,
    backend: str = config.DEFAULT_BACKEND,
    frame_index=None,
    class_index=None,
    conn=None,
    top_k: int = 100,
    class_top_k: int = config.DEFAULT_CLASS_TOP_K,
    rrf_k: int = config.RRF_K,
    leg_weights: dict | None = None,
    video_id: str | None = None,
) -> list:
    """Returns up to top_k results, best first:
    [{global_id, video_id, filename, pts_time, frame_idx, n, rrf_score,
      signals, thumbnail_path}, ...]
    `n` is the 1-indexed frame number matching the source filename (e.g.
    n=207 -> "207.jpg"). `signals` is {leg_name: {"rank": int, "contribution":
    float}} -- which signal(s) contributed to this result's rank, for UI
    debugging. `video_id`: restrict every leg to that one video (e.g. the
    UI's "search in this video only")."""
    owns_conn = conn is None
    if frame_index is None and class_index is None and conn is None:
        frame_index, class_index, conn = load_backend(backend)

    encode_text = backend_registry.get(backend)["encode_text"]
    query_vec = encode_text([query])
    faiss.normalize_L2(query_vec)

    id_range = None
    if video_id:
        video_row = store.get_video(conn, video_id)
        id_range = (video_row["start_global_id"], video_row["num_keyframes"]) if video_row else (0, 0)

    legs = {}
    frame_ids = _frame_leg(query_vec, frame_index, top_k, id_range=id_range)
    if frame_ids:
        legs["frame"] = frame_ids
    class_ids = _class_leg(query_vec, class_index, conn, class_top_k, top_k, id_range=id_range)
    if class_ids:
        legs["class"] = class_ids
    text_ids = _text_leg(query, conn, top_k, video_id=video_id)
    if text_ids:
        legs["text"] = text_ids

    fused = fusion.reciprocal_rank_fusion(legs, k=rrf_k, weights=leg_weights)[:top_k]
    kf_rows = store.get_keyframes_by_global_ids(conn, [gid for gid, _, _ in fused])

    results = []
    for gid, rrf_score, breakdown in fused:
        kf = kf_rows.get(gid)
        if kf is None:
            continue
        thumb = config.THUMBNAIL_ROOT / kf["video_id"] / kf["filename"]
        results.append({
            "global_id": gid,
            "video_id": kf["video_id"],
            "filename": kf["filename"],
            "pts_time": kf["pts_time"],
            "frame_idx": kf["frame_idx"],
            "n": kf["n"],
            "rrf_score": rrf_score,
            "signals": breakdown,
            "thumbnail_path": str(thumb) if thumb.exists() else None,
        })

    if owns_conn:
        conn.close()
    return results


def get_nearby_frames(conn, video_id: str, global_id: int, window: int = config.NEARBY_FRAMES_WINDOW) -> list:
    """Up to `window` frames on each side of global_id within the same
    video, ordered by position -- for the UI's "nearby frames" strip when a
    result is clicked."""
    frames = store.get_keyframes_by_video(conn, video_id)
    idx = next((i for i, f in enumerate(frames) if f["global_id"] == global_id), None)
    if idx is None:
        return []
    lo, hi = max(0, idx - window), min(len(frames), idx + window + 1)
    out = []
    for f in frames[lo:hi]:
        thumb = config.THUMBNAIL_ROOT / f["video_id"] / f["filename"]
        out.append({
            "global_id": f["global_id"],
            "filename": f["filename"],
            "n": f["n"],
            "pts_time": f["pts_time"],
            "is_target": f["global_id"] == global_id,
            "thumbnail_path": str(thumb) if thumb.exists() else None,
        })
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KIS-style text query against the C1 baseline index.")
    parser.add_argument("query")
    parser.add_argument("--backend", choices=list(config.BACKENDS.keys()), default=config.DEFAULT_BACKEND)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--class-top-k", type=int, default=config.DEFAULT_CLASS_TOP_K)
    parser.add_argument("--class-weight", type=float, default=config.DEFAULT_LEG_WEIGHTS["class"])
    parser.add_argument("--video-id", default=None, help="Restrict search to one video_id.")
    args = parser.parse_args()

    weights = {**config.DEFAULT_LEG_WEIGHTS, "class": args.class_weight}

    for r in search(
        args.query, backend=args.backend, top_k=args.top_k, class_top_k=args.class_top_k,
        leg_weights=weights, video_id=args.video_id,
    ):
        signals = ",".join(r["signals"].keys())
        ts = f"{r['pts_time']:.2f}s" if r["pts_time"] is not None else "?"
        frame_id = f"{r['n']:03d}" if r["n"] is not None else r["filename"]
        print(f"{r['rrf_score']:.4f}  {r['video_id']}  frame {frame_id}  @ {ts}  [{signals}]")
