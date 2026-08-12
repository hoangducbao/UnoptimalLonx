"""
retrieve.py — query-side KIS/VQA/TRAKE search against the store built by
index_pipeline.py (clip_features_flat_ip.index + aic_metadata.db).

This is the direct replacement for query_test.py's single hardcoded query:
loads the real index once, encodes a text query via text_encoder.py, and
joins FAISS hits back to keyframe/video metadata via store.py. VQA and
TRAKE search are added in later phases (Phase 2/3 of the approved plan);
only kis_search is implemented so far.
"""

import argparse
from pathlib import Path

import faiss

import store
import submission
import text_encoder

HERE = Path(__file__).resolve().parent
INDEX_DIR = HERE.parent / "index"


def load_backend(faiss_path: Path, db_path: Path):
    """Open the FAISS index + SQLite connection once; reuse across queries
    (this is what a future UI/batch-query runner should hold onto rather
    than reloading per query)."""
    index = faiss.read_index(str(faiss_path))
    conn = store.connect(db_path)
    return index, conn


def kis_search(query: str, index, conn, k: int = 100, method: str = "mclip") -> list:
    """Text query -> ranked list of dicts (best first), each:
    {video_id, frame_idx, n, pts_time, score, title}.
    score is cosine similarity (IndexFlatIP over L2-normalized vectors)."""
    qvec = text_encoder.encode_text_query(query, method=method)
    scores, global_ids = index.search(qvec, k)
    scores, global_ids = scores[0], global_ids[0]

    valid = [(int(gid), float(score)) for gid, score in zip(global_ids, scores) if gid != -1]
    rows = store.get_keyframes_by_global_ids(conn, [gid for gid, _ in valid])

    results = []
    for gid, score in valid:
        row = rows.get(gid)
        if row is None:
            continue  # shouldn't happen if index/store are in sync, but don't crash a query over it
        results.append(
            {
                "global_id": gid,
                "video_id": row["video_id"],
                "n": row["n"],
                "frame_idx": row["frame_idx"],
                "pts_time": row["pts_time"],
                "score": score,
                "title": row["title"],
            }
        )
    return results


def _print_results(query: str, method: str, results: list, limit: int = 10) -> None:
    print(f"\nSearch results for: '{query}' (method={method})")
    print("=" * 70)
    for rank, r in enumerate(results[:limit], start=1):
        print(f"Rank {rank}: score={r['score']:.4f}")
        print(f"  video_id={r['video_id']}  frame_idx={r['frame_idx']}  pts_time={r['pts_time']:.2f}s")
        print(f"  title: {r['title'] or '(no media-info)'}")
        print("-" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the AICPrep retrieval store.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    kis_parser = subparsers.add_parser("kis", help="Known-Item Search: text -> ranked (video_id, frame_id)")
    kis_parser.add_argument("query", type=str)
    kis_parser.add_argument("--k", type=int, default=100)
    kis_parser.add_argument("--method", choices=["mclip", "translate"], default="mclip")
    kis_parser.add_argument("--faiss-path", type=Path, default=INDEX_DIR / "clip_features_flat_ip.index")
    kis_parser.add_argument("--db-path", type=Path, default=INDEX_DIR / "aic_metadata.db")
    kis_parser.add_argument("--out", type=Path, default=None, help="Write submission CSV here")
    kis_parser.add_argument("--show", type=int, default=10, help="How many top results to print")

    args = parser.parse_args()

    if args.command == "kis":
        index, conn = load_backend(args.faiss_path, args.db_path)
        results = kis_search(args.query, index, conn, k=args.k, method=args.method)
        _print_results(args.query, args.method, results, limit=args.show)
        if args.out:
            submission.write_kis(results, args.out)
            print(f"\nWrote {min(len(results), 100)} rows -> {args.out}")
        conn.close()
