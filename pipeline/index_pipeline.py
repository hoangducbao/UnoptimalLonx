"""
index_pipeline.py — Step 3: build/extend the searchable index from
loader.py + clean_objects.py output.

Join point between the FAISS side (vector similarity search) and the
metadata side (store.py's SQLite), keyed by a `global_id` that increments
once per keyframe across the ENTIRE dataset and never resets per video —
FAISS row `i` and SQLite `keyframes.global_id = i` refer to the same
keyframe by construction (vectors are appended to FAISS and rows inserted
into SQLite in the same order, in the same per-video transaction).

Two modes:
  --incremental (default): diff on-disk video_ids against store.py's
      "already known" set (indexed OR skipped) and process only the new
      ones, appending to the existing FAISS index + DB. This is what lets
      a weekly-arriving batch get indexed without re-touching Batch 1.
  --rebuild: delete any existing index/db and start over from scratch.

Videos that fail loader.py's invariant checks (npy/csv count mismatch,
missing object json) are flagged into skipped_videos and skipped rather
than aborting the whole run.
"""

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

import clean_objects
import loader
import store

HERE = Path(__file__).resolve().parent
INDEX_DIR = HERE.parent / "index"

logger = logging.getLogger(__name__)

VECTOR_DIM = 512


@dataclass
class IndexedKeyframe:
    global_id: int
    video_id: str
    n: int
    pts_time: float
    frame_idx: int
    image_path: str


@dataclass
class IndexedVideo:
    video_id: str
    start_global_id: int
    num_keyframes: int
    title: str | None
    author: str | None
    watch_url: str | None
    description: str | None
    publish_date: str | None
    length: int | None
    has_media_info: bool
    indexed_at: str


def _build_indexed_video(result: loader.VideoLoadResult, start_global_id: int) -> IndexedVideo:
    meta = result.video_meta
    return IndexedVideo(
        video_id=result.video_id,
        start_global_id=start_global_id,
        num_keyframes=len(result.records),
        title=meta.get("title"),
        author=meta.get("author"),
        watch_url=meta.get("watch_url"),
        description=meta.get("description"),
        publish_date=meta.get("publish_date"),
        length=meta.get("length"),
        has_media_info=result.has_media_info,
        indexed_at=datetime.now(timezone.utc).isoformat(),
    )


def _build_indexed_keyframes(records: list, start_global_id: int) -> list:
    return [
        IndexedKeyframe(
            global_id=start_global_id + i,
            video_id=r.video_id,
            n=r.n,
            pts_time=r.pts_time,
            frame_idx=r.frame_idx,
            image_path=r.image_path,
        )
        for i, r in enumerate(records)
    ]


@dataclass
class IndexingReport:
    video_ids_total: list
    video_ids_processed: list
    video_ids_skipped: list  # list[(video_id, reason)]
    keyframes_indexed_this_run: int
    total_keyframes: int
    faiss_path: Path
    db_path: Path


def run_indexing_pipeline(
    dataset_root: Path,
    faiss_path: Path,
    db_path: Path,
    incremental: bool = True,
    video_ids: list | None = None,
    require_objects: bool = True,
    min_confidence: float = 0.3,
    max_labels: int | None = None,
    vector_dim: int = VECTOR_DIM,
    io_workers: int = loader.DEFAULT_IO_WORKERS,
    progress_every: int = 10,
) -> IndexingReport:
    dataset_root = Path(dataset_root)
    faiss_path = Path(faiss_path)
    db_path = Path(db_path)

    if not incremental:
        for p in (faiss_path, db_path):
            if p.exists():
                p.unlink()
        for wal_suffix in ("-wal", "-shm"):
            wal_path = Path(str(db_path) + wal_suffix)
            if wal_path.exists():
                wal_path.unlink()

    all_video_ids = video_ids if video_ids is not None else loader.discover_video_ids(dataset_root)

    conn = store.connect(db_path)
    index = faiss.read_index(str(faiss_path)) if faiss_path.exists() else faiss.IndexFlatIP(vector_dim)

    already_known = store.get_known_video_ids(conn) if incremental else set()
    to_process = [v for v in all_video_ids if v not in already_known]

    processed: list = []
    skipped: list = []
    next_global_id = store.next_global_id(conn)
    start_global_id_this_run = next_global_id
    assert next_global_id == index.ntotal, (
        f"store/index out of sync: SQLite has {next_global_id} keyframes, "
        f"FAISS index has {index.ntotal} vectors. Re-run with --rebuild."
    )

    t_start = time.monotonic()

    try:
        for i, video_id in enumerate(to_process):
            try:
                result = loader.load_video(
                    video_id, dataset_root, require_objects=require_objects, io_workers=io_workers
                )
            except (AssertionError, FileNotFoundError) as e:
                logger.warning("Skipping video_id=%s: %s", video_id, e)
                reason = str(e)
                store.insert_skipped_video(conn, video_id, reason, datetime.now(timezone.utc).isoformat())
                conn.commit()
                skipped.append((video_id, reason))
                continue

            if not result.records:
                logger.warning("Skipping video_id=%s: zero keyframes", video_id)
                store.insert_skipped_video(conn, video_id, "zero keyframes", datetime.now(timezone.utc).isoformat())
                conn.commit()
                skipped.append((video_id, "zero keyframes"))
                continue

            cleaned = clean_objects.clean_records(
                result.records, min_confidence=min_confidence, max_labels=max_labels
            )
            indexed_kfs = _build_indexed_keyframes(result.records, next_global_id)
            indexed_video = _build_indexed_video(result, next_global_id)

            vecs = np.vstack([r.clip_vector for r in result.records]).astype("float32")
            faiss.normalize_L2(vecs)
            index.add(vecs)

            store.insert_video(conn, indexed_video)
            store.insert_keyframes(conn, indexed_kfs)
            store.insert_keyframe_text(
                conn,
                [(c.text, "objects", video_id, kf.global_id) for c, kf in zip(cleaned, indexed_kfs)],
            )
            # NOT committed here on purpose -- see checkpoint below. Committing
            # every video would let SQLite's on-disk state race ahead of the
            # FAISS index (only written to disk at checkpoints), so a kill
            # between checkpoints would leave the two stores out of sync
            # (exactly what happened during an earlier interrupted run). The
            # pending SQLite writes for videos since the last checkpoint are
            # rolled back automatically by SQLite/WAL if the process dies
            # before the matching commit below runs.

            next_global_id += len(result.records)
            processed.append(video_id)

            done = i + 1
            is_checkpoint = progress_every and done % progress_every == 0
            is_last = done == len(to_process)
            if is_checkpoint or is_last:
                conn.commit()
                faiss.write_index(index, str(faiss_path))
                elapsed = time.monotonic() - t_start
                rate = done / elapsed if elapsed > 0 else 0.0
                remaining = len(to_process) - done
                eta_s = remaining / rate if rate > 0 else 0.0
                logger.info(
                    "Checkpoint: %d/%d videos (%d keyframes so far), %.1f videos/s, ETA %.0fs",
                    done, len(to_process), next_global_id, rate, eta_s,
                )
    finally:
        conn.close()

    return IndexingReport(
        video_ids_total=list(all_video_ids),
        video_ids_processed=processed,
        video_ids_skipped=skipped,
        keyframes_indexed_this_run=next_global_id - start_global_id_this_run,
        total_keyframes=next_global_id,
        faiss_path=faiss_path,
        db_path=db_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Build/extend the FAISS index + SQLite metadata store for the AIC dataset."
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--faiss-path", type=Path, default=INDEX_DIR / "clip_features_flat_ip.index")
    parser.add_argument("--db-path", type=Path, default=INDEX_DIR / "aic_metadata.db")
    parser.add_argument("--rebuild", action="store_true", help="Delete existing index/db and start over")
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--max-labels", type=int, default=None)
    parser.add_argument(
        "--video-ids", nargs="+", default=None,
        help="Index only these video_ids instead of discovering every video under "
        "dataset_root/clip-features-32/. Useful for a quick test run.",
    )
    parser.add_argument(
        "--allow-missing-objects", action="store_true",
        help="Don't require every keyframe to have an object-detection json.",
    )
    parser.add_argument(
        "--io-workers", type=int, default=loader.DEFAULT_IO_WORKERS,
        help="Thread count for reading per-keyframe object-detection json (I/O-bound; "
        "the dominant cost at full-corpus scale). Set 1 to force serial reads.",
    )
    parser.add_argument(
        "--progress-every", type=int, default=10,
        help="Log progress and checkpoint (commit + write index) every N videos. "
        "0 disables checkpointing (only writes at the very end).",
    )
    args = parser.parse_args()

    report = run_indexing_pipeline(
        dataset_root=args.dataset_root,
        faiss_path=args.faiss_path,
        db_path=args.db_path,
        incremental=not args.rebuild,
        video_ids=args.video_ids,
        require_objects=not args.allow_missing_objects,
        min_confidence=args.min_confidence,
        max_labels=args.max_labels,
        io_workers=args.io_workers,
        progress_every=args.progress_every,
    )
    logger.info(
        "Indexed %d new keyframes from %d/%d videos (%d skipped) -> %s, %s. Total keyframes now: %d",
        report.keyframes_indexed_this_run,
        len(report.video_ids_processed),
        len(report.video_ids_total),
        len(report.video_ids_skipped),
        report.faiss_path,
        report.db_path,
        report.total_keyframes,
    )
    for video_id, reason in report.video_ids_skipped:
        logger.warning("Skipped %s: %s", video_id, reason)
