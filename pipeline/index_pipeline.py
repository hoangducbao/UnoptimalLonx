"""
index_pipeline.py — Step 3: build the FAISS indices + SQLite metadata store
from loader.py + detections.py output.

Two FAISS indices, both faiss.IndexFlatIP(768):
  frame index  -- one vector per keyframe (config.FRAME_FAISS_PATH)
  class index  -- one vector per distinct object-detection class label,
                   embedded via ViCLIP-OT's text tower (config.CLASS_FAISS_PATH)

global_id increments once per keyframe across the entire run and never
resets per video -- frame FAISS row `i` and SQLite `keyframes.global_id = i`
refer to the same keyframe by construction (vectors appended to FAISS and
rows inserted into SQLite in the same order, in the same per-video batch).

Per spec ("wipe-and-rebuild is acceptable" for this dev scaffold), the
default -- and only -- mode is a full rebuild every run: simpler than the
incremental-diff logic in old_version/pipeline/index_pipeline.py.
`--video-ids`/`--limit` subset the run for fast dev iteration.

Videos that fail loader.py's invariant checks are flagged into
skipped_videos and skipped rather than aborting the whole run.
"""

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

import config
import detections
import loader
import store
import viclip_encoder

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
    video_ids_total: list
    video_ids_processed: list
    video_ids_skipped: list  # list[(video_id, reason)]
    keyframes_indexed: int
    classes_indexed: int
    frame_faiss_path: Path
    class_faiss_path: Path
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
    dataset_root: Path = config.FRAME_DATA_ROOT,
    timestamp_root: Path = config.TIMESTAMP_ROOT,
    detections_csv: Path = config.DETECTIONS_CSV,
    frame_faiss_path: Path = config.FRAME_FAISS_PATH,
    class_faiss_path: Path = config.CLASS_FAISS_PATH,
    db_path: Path = config.DB_PATH,
    video_ids: list | None = None,
    limit: int | None = None,
    progress_every: int = 10,
) -> IndexingReport:
    frame_faiss_path = Path(frame_faiss_path)
    class_faiss_path = Path(class_faiss_path)
    db_path = Path(db_path)
    frame_faiss_path.parent.mkdir(parents=True, exist_ok=True)

    _wipe(frame_faiss_path, class_faiss_path, db_path)

    all_video_ids = (
        video_ids if video_ids is not None
        else loader.discover_video_ids(dataset_root / "embeddings")
    )
    if limit is not None:
        all_video_ids = all_video_ids[:limit]

    conn = store.connect(db_path)
    frame_index = faiss.IndexFlatIP(config.VECTOR_DIM)

    logger.info("Loading object detections from %s", detections_csv)
    det_store = detections.load_detections(detections_csv)

    next_gid = 0
    processed, skipped = [], []
    t_start = time.monotonic()

    for i, video_id in enumerate(all_video_ids):
        try:
            result = loader.load_video(video_id, dataset_root, timestamp_root)
        except (AssertionError, FileNotFoundError) as e:
            logger.warning("Skipping video_id=%s: %s", video_id, e)
            store.insert_skipped_video(conn, video_id, str(e), datetime.now(timezone.utc).isoformat())
            conn.commit()
            skipped.append((video_id, str(e)))
            continue

        if not result.records:
            logger.warning("Skipping video_id=%s: zero keyframes", video_id)
            store.insert_skipped_video(conn, video_id, "zero keyframes", datetime.now(timezone.utc).isoformat())
            conn.commit()
            skipped.append((video_id, "zero keyframes"))
            continue

        start_gid = next_gid
        indexed_kfs = [
            IndexedKeyframe(
                global_id=start_gid + r.row_index, video_id=r.video_id, row_index=r.row_index,
                filename=r.filename, pts_time=r.pts_time, fps=r.fps, frame_idx=r.frame_idx, n=r.n,
            )
            for r in result.records
        ]
        indexed_video = IndexedVideo(
            video_id=video_id, start_global_id=start_gid, num_keyframes=len(result.records),
            has_timestamps=result.has_timestamps, indexed_at=datetime.now(timezone.utc).isoformat(),
        )

        vecs = np.vstack([r.embedding for r in result.records]).astype("float32")
        faiss.normalize_L2(vecs)
        frame_index.add(vecs)

        frame_class_rows = []
        for r, kf in zip(result.records, indexed_kfs):
            frame_number = r.n if r.n is not None else r.row_index + 1
            for class_name, score in det_store.frame_classes(video_id, frame_number).items():
                frame_class_rows.append((kf.global_id, class_name, score))

        store.insert_video(conn, indexed_video)
        store.insert_keyframes(conn, indexed_kfs)
        if frame_class_rows:
            store.insert_frame_classes(conn, frame_class_rows)
        # NOT committed here on purpose -- checkpointed below in lockstep
        # with the FAISS write, same reasoning as old_version's
        # index_pipeline.py: committing every video would let SQLite's
        # on-disk state race ahead of the FAISS index (only flushed to disk
        # at checkpoints), so a kill between checkpoints could leave the two
        # stores out of sync.

        next_gid += len(result.records)
        processed.append(video_id)

        done = i + 1
        is_checkpoint = progress_every and done % progress_every == 0
        is_last = done == len(all_video_ids)
        if is_checkpoint or is_last:
            conn.commit()
            faiss.write_index(frame_index, str(frame_faiss_path))
            elapsed = time.monotonic() - t_start
            rate = done / elapsed if elapsed > 0 else 0.0
            logger.info(
                "Checkpoint: %d/%d videos (%d keyframes so far), %.1f videos/s",
                done, len(all_video_ids), next_gid, rate,
            )

    # Class leg: embed the distinct class vocabulary once, after all videos,
    # so the class FAISS index and its SQLite class_row_id mapping are built
    # from the exact same ordered list in one shot.
    classes_indexed = 0
    if det_store.classes:
        logger.info("Embedding %d distinct object classes via ViCLIP-OT text tower", len(det_store.classes))
        class_vecs = viclip_encoder.encode_text(det_store.classes)
        faiss.normalize_L2(class_vecs)
        class_index = faiss.IndexFlatIP(config.VECTOR_DIM)
        class_index.add(class_vecs)
        faiss.write_index(class_index, str(class_faiss_path))
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
        video_ids_total=list(all_video_ids), video_ids_processed=processed, video_ids_skipped=skipped,
        keyframes_indexed=next_gid, classes_indexed=classes_indexed,
        frame_faiss_path=frame_faiss_path, class_faiss_path=class_faiss_path, db_path=db_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Build the frame + object-class FAISS indices and SQLite metadata store "
                     "for the C1 baseline. Always a full rebuild (wipe-and-rebuild is fine for "
                     "this dev scaffold) -- use --video-ids/--limit to iterate on a subset."
    )
    parser.add_argument("--dataset-root", type=Path, default=config.FRAME_DATA_ROOT)
    parser.add_argument("--timestamp-root", type=Path, default=config.TIMESTAMP_ROOT)
    parser.add_argument("--detections-csv", type=Path, default=config.DETECTIONS_CSV)
    parser.add_argument("--video-ids", nargs="+", default=None,
                         help="Index only these video_ids instead of discovering every video.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Index only the first N discovered/given video_ids.")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    report = run_indexing_pipeline(
        dataset_root=args.dataset_root, timestamp_root=args.timestamp_root,
        detections_csv=args.detections_csv, video_ids=args.video_ids, limit=args.limit,
        progress_every=args.progress_every,
    )
    logger.info(
        "Indexed %d keyframes from %d/%d videos (%d skipped), %d object classes -> %s, %s, %s",
        report.keyframes_indexed, len(report.video_ids_processed), len(report.video_ids_total),
        len(report.video_ids_skipped), report.classes_indexed,
        report.frame_faiss_path, report.class_faiss_path, report.db_path,
    )
    for video_id, reason in report.video_ids_skipped:
        logger.warning("Skipped %s: %s", video_id, reason)
