"""
retrieve.py — query-time pipeline (spec's 5 steps):

  1. encode query with ViCLIP-OT, search the frame embedding index
  2. encode query with ViCLIP-OT, search the object-class embedding index,
     map matched classes back to frames via the frame_classes join table
  3. search OCR/caption text via FTS5 (trigram, diacritic-folded)
  4. fuse the (up to three) ranked lists with RRF (fusion.py)
  5. return ranked frame results with video/timestamp/thumbnail info

Each leg degrades gracefully: a leg with no data (empty/missing index,
empty FTS5 table) is simply omitted from fusion rather than raising.
"""

import argparse
from pathlib import Path

import faiss

import config
import fusion
import store
import viclip_encoder


def load_backend(
    frame_faiss_path: Path = config.FRAME_FAISS_PATH,
    class_faiss_path: Path = config.CLASS_FAISS_PATH,
    db_path: Path = config.DB_PATH,
):
    """Open the frame/class FAISS indices (None if not yet built -- that leg
    then degrades gracefully) and the metadata DB."""
    conn = store.connect(db_path)
    frame_index = faiss.read_index(str(frame_faiss_path)) if Path(frame_faiss_path).exists() else None
    class_index = faiss.read_index(str(class_faiss_path)) if Path(class_faiss_path).exists() else None
    return frame_index, class_index, conn


def _frame_leg(query_vec, frame_index, top_k: int) -> list:
    if frame_index is None or frame_index.ntotal == 0:
        return []
    _scores, ids = frame_index.search(query_vec, min(top_k, frame_index.ntotal))
    return [int(i) for i in ids[0] if i != -1]


def _class_leg(query_vec, class_index, conn, top_classes: int, top_k: int) -> list:
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
        if gid not in best_frame_score or match_score > best_frame_score[gid]:
            best_frame_score[gid] = match_score

    ranked = sorted(best_frame_score.items(), key=lambda kv: kv[1], reverse=True)
    return [gid for gid, _ in ranked[:top_k]]


def _text_leg(query: str, conn, top_k: int) -> list:
    if conn.execute("SELECT COUNT(*) AS n FROM keyframe_text").fetchone()["n"] == 0:
        return []  # no OCR/caption data yet -- skip rather than run an empty-table query
    return [row["global_id"] for row in store.search_text(conn, query, limit=top_k)]


def search(
    query: str,
    frame_index=None,
    class_index=None,
    conn=None,
    top_k: int = 100,
    class_top_k: int = config.DEFAULT_CLASS_TOP_K,
    rrf_k: int = config.RRF_K,
    leg_weights: dict | None = None,
) -> list:
    """Returns up to top_k results, best first:
    [{global_id, video_id, filename, pts_time, frame_idx, n, rrf_score,
      signals, thumbnail_path}, ...]
    `n` is the 1-indexed frame number matching the source filename (e.g.
    n=207 -> "207.jpg"). `signals` is {leg_name: {"rank": int, "contribution":
    float}} -- which signal(s) contributed to this result's rank, for UI
    debugging. `leg_weights`: see fusion.reciprocal_rank_fusion."""
    owns_conn = conn is None
    if frame_index is None and class_index is None and conn is None:
        frame_index, class_index, conn = load_backend()

    query_vec = viclip_encoder.encode_text([query])
    faiss.normalize_L2(query_vec)

    legs = {}
    frame_ids = _frame_leg(query_vec, frame_index, top_k)
    if frame_ids:
        legs["frame"] = frame_ids
    class_ids = _class_leg(query_vec, class_index, conn, class_top_k, top_k)
    if class_ids:
        legs["class"] = class_ids
    text_ids = _text_leg(query, conn, top_k)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KIS-style text query against the C1 baseline index.")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--class-top-k", type=int, default=config.DEFAULT_CLASS_TOP_K)
    parser.add_argument("--class-weight", type=float, default=config.DEFAULT_LEG_WEIGHTS["class"])
    args = parser.parse_args()

    weights = {**config.DEFAULT_LEG_WEIGHTS, "class": args.class_weight}

    for r in search(args.query, top_k=args.top_k, class_top_k=args.class_top_k, leg_weights=weights):
        signals = ",".join(r["signals"].keys())
        ts = f"{r['pts_time']:.2f}s" if r["pts_time"] is not None else "?"
        frame_id = f"{r['n']:03d}" if r["n"] is not None else r["filename"]
        print(f"{r['rrf_score']:.4f}  {r['video_id']}  frame {frame_id}  @ {ts}  [{signals}]")
