"""
loader.py — Step 1: load one video's ViCLIP-OT frame embeddings, joined with
timestamps when available.

Reads AICDataExtracted/embeddings/{video_id}_viclip768.npy (N, 768) plus its
sibling _filenames.csv (row_index, filename), and, best-effort, joins by row
index against AICData/map-keyframes/{video_id}.csv (n, pts_time, fps,
frame_idx) -- verified during planning to line up 1:1, in order, with the
new embeddings for every video checked.

A missing/misaligned map-keyframes file is tolerated (per spec: "tolerate
partial data") -- the video is still indexed, just with null timestamps. A
row-count mismatch between the .npy and its own _filenames.csv is NOT
tolerated: that would mean the frame vectors and their filenames are already
out of sync in the source data, which is a real correctness invariant.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


@dataclass
class KeyframeRecord:
    video_id: str
    row_index: int          # 0-indexed, == npy row == filenames.csv row_index
    filename: str            # e.g. "001.jpg"
    embedding: np.ndarray     # (768,) float32
    pts_time: float | None
    fps: float | None
    frame_idx: int | None    # competition-submission frame index, if known
    n: int | None            # 1-indexed keyframe number from map-keyframes, if known


@dataclass
class VideoLoadResult:
    video_id: str
    records: list
    has_timestamps: bool


def discover_video_ids(embeddings_dir: Path = config.EMBEDDINGS_DIR) -> list:
    return sorted(
        p.name[: -len("_viclip768.npy")]
        for p in embeddings_dir.glob("*_viclip768.npy")
    )


def _load_timestamps(video_id: str, timestamp_root: Path, expected_rows: int):
    """Returns a list of (pts_time, fps, frame_idx, n) aligned 1:1 with
    embedding row_index, or None if no usable timestamp file was found."""
    csv_path = timestamp_root / f"{video_id}.csv"
    if not csv_path.exists():
        return None
    ts = pd.read_csv(csv_path)
    if len(ts) != expected_rows:
        logger.warning(
            "video_id=%s: map-keyframes row count (%d) != embeddings row count (%d); "
            "indexing without timestamps",
            video_id, len(ts), expected_rows,
        )
        return None
    ts = ts.sort_values("n").reset_index(drop=True)
    return list(zip(ts["pts_time"], ts["fps"], ts["frame_idx"], ts["n"]))


def load_video(
    video_id: str,
    dataset_root: Path = config.FRAME_DATA_ROOT,
    timestamp_root: Path = config.TIMESTAMP_ROOT,
) -> VideoLoadResult:
    embeddings_dir = dataset_root / "embeddings"
    npy_path = embeddings_dir / f"{video_id}_viclip768.npy"
    filenames_path = embeddings_dir / f"{video_id}_viclip768_filenames.csv"

    if not npy_path.exists() or not filenames_path.exists():
        raise FileNotFoundError(f"missing embeddings for video_id={video_id!r}")

    vectors = np.load(npy_path)
    filenames_df = pd.read_csv(filenames_path).sort_values("row_index").reset_index(drop=True)

    if len(filenames_df) != vectors.shape[0]:
        raise AssertionError(
            f"video_id={video_id!r}: npy has {vectors.shape[0]} rows but "
            f"filenames.csv has {len(filenames_df)} rows"
        )

    timestamps = _load_timestamps(video_id, timestamp_root, expected_rows=vectors.shape[0])

    records = []
    for row_index, filename in enumerate(filenames_df["filename"]):
        if timestamps is not None:
            pts_time, fps, frame_idx, n = timestamps[row_index]
            pts_time, fps, frame_idx, n = float(pts_time), float(fps), int(frame_idx), int(n)
        else:
            pts_time = fps = frame_idx = n = None
        records.append(
            KeyframeRecord(
                video_id=video_id,
                row_index=row_index,
                filename=filename,
                embedding=vectors[row_index].astype("float32"),
                pts_time=pts_time,
                fps=fps,
                frame_idx=frame_idx,
                n=n,
            )
        )

    return VideoLoadResult(video_id=video_id, records=records, has_timestamps=timestamps is not None)
