"""
index_pipeline.py — Step 3: build the FAISS indices + SQLite metadata store
from loader.py + detections.py output, for one or both selectable
embedding backends (config.BACKENDS / backends.py).

Both backends index the exact same keyframes, in the exact same per-video
order (verified: all 873 videos match row-for-row between
AICDataExtracted/embeddings and AICData/clip-features-32), so a single
SQLite metadata store and a single global_id numbering scheme serve both --
only each backend's own frame + class FAISS files differ. A video is only
indexed if EVERY requested backend can load it (and all agree on keyframe
count), so global_id stays a valid row index into every requested backend's
frame FAISS index by construction.

Per spec ("wipe-and-rebuild is acceptable" for this dev scaffold), the
default -- and only -- mode is a full rebuild every run: simpler than the
incremental-diff logic in old_version/pipeline/index_pipeline.py.
`--video-ids`/`--limit` subset the run for fast dev iteration; `--backends`
limits which backend(s) get (re)built this run (still requires the DB to be
rebuilt from scratch, so omitting a previously-built backend here means
re-running with it included later to get it back).

Videos that fail any requested backend's loader invariant checks are
flagged into skipped_videos and skipped rather than aborting the whole run.
"""

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

import backends as backend_registry
import config
import detections
import loader
import store

logger = logging.getLogger(__name__)


@dataclass
class IndexedKeyframe:
    global_id: int
    video_id: str
    row_index: int
    filename: str
    pts_time: float | None
    fps: float | None
    frame_idx: int | None
    n: int | None


@dataclass
class IndexedVideo:
    video_id: str
    start_global_id: int
    num_keyframes: int
    has_timestamps: bool
    indexed_at: str


@dataclass
class IndexingReport:
    backends: list
    video_ids_total: list
    video_ids_processed: list
    video_ids_skipped: list  # list[(video_id, reason)]
    keyframes_indexed: int
    classes_indexed: int
    db_path: Path


def _wipe(*paths) -> None:
    for p in paths:
        p = Path(p)
        if p.exists():
            p.unlink()
        for suffix in ("-wal", "-shm"):
            wal = Path(str(p) + suffix)
            if wal.exists():
                wal.unlink()


def run_indexing_pipeline(
    backends: list | None = None,
    timestamp_root: Path = config.TIMESTAMP_ROOT,
    detections_csv: Path = config.DETECTIONS_CSV,
    db_path: Path = config.DB_PATH,
    video_ids: list | None = None,
    limit: int | None = None,
    progress_every: int = 10,
) -> IndexingReport:
    backend_keys = backends if backends is not None else list(config.BACKENDS.keys())
    backend_defs = {b: backend_registry.get(b) for b in backend_keys}

    db_path = Path(db_path)
    frame_faiss_paths = {b: Path(d["frame_faiss_path"]) for b, d in backend_defs.items()}
    class_faiss_paths = {b: Path(d["class_faiss_path"]) for b, d in backend_defs.items()}
    db_path.parent.mkdir(parents=True, exist_ok=True)

    _wipe(db_path, *frame_faiss_paths.values(), *class_faiss_paths.values())

    all_video_ids = (
        video_ids if video_ids is not None
        else loader.discover_video_ids(config.EMBEDDINGS_DIR)
    )
    if limit is not None:
        all_video_ids = all_video_ids[:limit]

    conn = store.connect(db_path)
    frame_indices = {b: faiss.IndexFlatIP(d["dim"]) for b, d in backend_defs.items()}

    logger.info("Loading object detections from %s", detections_csv)
    det_store = detections.load_detections(detections_csv)

    next_gid = 0
    processed, skipped = [], []
    t_start = time.monotonic()

    def _skip(video_id: str, reason: str) -> None:
        logger.warning("Skipping video_id=%s: %s", video_id, reason)
        store.insert_skipped_video(conn, video_id, reason, datetime.now(timezone.utc).isoformat())
        conn.commit()
        skipped.append((video_id, reason))

    for i, video_id in enumerate(all_video_ids):
        try:
            per_backend = {b: d["load_video"](video_id, timestamp_root=timestamp_root)
                           for b, d in backend_defs.items()}
        except (AssertionError, FileNotFoundError) as e:
            _skip(video_id, str(e))
            continue

        counts = {b: len(r.records) for b, r in per_backend.items()}
        if len(set(counts.values())) > 1:
            _skip(video_id, f"backend keyframe-count mismatch: {counts}")
            continue

        reference = next(iter(per_backend.values()))
        if not reference.records:
            _skip(video_id, "zero keyframes")
            continue

        start_gid = next_gid
        indexed_kfs = [
            IndexedKeyframe(
                global_id=start_gid + r.row_index, video_id=r.video_id, row_index=r.row_index,
                filename=r.filename, pts_time=r.pts_time, fps=r.fps, frame_idx=r.frame_idx, n=r.n,
            )
            for r in reference.records
        ]
        indexed_video = IndexedVideo(
            video_id=video_id, start_global_id=start_gid, num_keyframes=len(reference.records),
            has_timestamps=reference.has_timestamps, indexed_at=datetime.now(timezone.utc).isoformat(),
        )

        for b, result in per_backend.items():
            vecs = np.vstack([r.embedding for r in result.records]).astype("float32")
            faiss.normalize_L2(vecs)
            frame_indices[b].add(vecs)

        frame_class_rows = []
        for r, kf in zip(reference.records, indexed_kfs):
            frame_number = r.n if r.n is not None else r.row_index + 1
            for class_name, score in det_store.frame_classes(video_id, frame_number).items():
                frame_class_rows.append((kf.global_id, class_name, score))

        store.insert_video(conn, indexed_video)
        store.insert_keyframes(conn, indexed_kfs)
        if frame_class_rows:
            store.insert_frame_classes(conn, frame_class_rows)
        # NOT committed here on purpose -- checkpointed below in lockstep
        # with every backend's FAISS write, same reasoning as
        # old_version's index_pipeline.py: committing every video would let
        # SQLite's on-disk state race ahead of the FAISS indices (only
        # flushed to disk at checkpoints).

        next_gid += len(reference.records)
        processed.append(video_id)

        done = i + 1
        is_checkpoint = progress_every and done % progress_every == 0
        is_last = done == len(all_video_ids)
        if is_checkpoint or is_last:
            conn.commit()
            for b, idx in frame_indices.items():
                faiss.write_index(idx, str(frame_faiss_paths[b]))
            elapsed = time.monotonic() - t_start
            rate = done / elapsed if elapsed > 0 else 0.0
            logger.info(
                "Checkpoint: %d/%d videos (%d keyframes so far), %.1f videos/s",
                done, len(all_video_ids), next_gid, rate,
            )

    # Class leg: embed the distinct class vocabulary once per backend, after
    # all videos, so each class FAISS index and the shared SQLite
    # class_row_id mapping are built from the exact same ordered list.
    classes_indexed = 0
    if det_store.classes:
        for b, d in backend_defs.items():
            logger.info("[%s] Embedding %d distinct object classes", b, len(det_store.classes))
            class_vecs = d["encode_text"](det_store.classes)
            faiss.normalize_L2(class_vecs)
            class_index = faiss.IndexFlatIP(d["dim"])
            class_index.add(class_vecs)
            faiss.write_index(class_index, str(class_faiss_paths[b]))
        store.insert_classes(conn, det_store.classes)
        conn.commit()
        classes_indexed = len(det_store.classes)
    else:
        logger.warning(
            "No object-detection classes found -- class FAISS index not built; "
            "the object-class leg will be skipped at query time"
        )

    conn.close()

    return IndexingReport(
        backends=list(backend_defs.keys()), video_ids_total=list(all_video_ids),
        video_ids_processed=processed, video_ids_skipped=skipped,
        keyframes_indexed=next_gid, classes_indexed=classes_indexed, db_path=db_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Build the frame + object-class FAISS indices and SQLite metadata store "
                     "for the C1 baseline. Always a full rebuild (wipe-and-rebuild is fine for "
                     "this dev scaffold) -- use --video-ids/--limit to iterate on a subset."
    )
    parser.add_argument("--backends", nargs="+", choices=list(config.BACKENDS.keys()), default=None,
                         help="Which backend(s) to build. Default: all of them.")
    parser.add_argument("--timestamp-root", type=Path, default=config.TIMESTAMP_ROOT)
    parser.add_argument("--detections-csv", type=Path, default=config.DETECTIONS_CSV)
    parser.add_argument("--video-ids", nargs="+", default=None,
                         help="Index only these video_ids instead of discovering every video.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Index only the first N discovered/given video_ids.")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    report = run_indexing_pipeline(
        backends=args.backends, timestamp_root=args.timestamp_root,
        detections_csv=args.detections_csv, video_ids=args.video_ids, limit=args.limit,
        progress_every=args.progress_every,
    )
    logger.info(
        "[%s] Indexed %d keyframes from %d/%d videos (%d skipped), %d object classes -> %s",
        ",".join(report.backends), report.keyframes_indexed, len(report.video_ids_processed),
        len(report.video_ids_total), len(report.video_ids_skipped), report.classes_indexed,
        report.db_path,
    )
    for video_id, reason in report.video_ids_skipped:
        logger.warning("Skipped %s: %s", video_id, reason)
