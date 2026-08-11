"""
loader.py — Step 1: load and join one video's per-source files into a flat
list of per-keyframe records.

Dataset layout expected under `dataset_root` (see AICData):
    clip-features-32/{video_id}.npy   CLIP ViT-B/32 vectors, shape (num_keyframes, 512), float16, L2-normalized
    map-keyframes/{video_id}.csv      columns: n, pts_time, fps, frame_idx (one row per keyframe)
    media-info/{video_id}.json        video-level YouTube metadata (title, description, ...) — MAY BE ABSENT
    objects/{video_id}/{n:03d}.json   per-keyframe object detections (n is 1-indexed)
    keyframes/{video_id}/{n:03d}.jpg  the actual keyframe image (referenced, not loaded)

This module only joins — it doesn't clean/filter/index anything. It raises
on the one invariant that actually matters (npy/csv row-count mismatch,
which would silently misalign every downstream frame_id); callers decide
what "raise" means for their run (index_pipeline.py catches it and skips
the video rather than aborting the whole batch).

Missing media-info is NOT an error — some videos genuinely lack it per the
organizer's own docs — `video_meta` is just `{}` in that case, and callers
must treat metadata fields as optional display sugar, never required.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Reading one keyframe's object-detection json is a tiny amount of CPU work
# behind a real filesystem round-trip (open+read+close, often also an AV
# scan on Windows) -- at ~200 keyframes/video x 873 videos = ~177K of these,
# doing them one at a time serializes on I/O wait, not CPU. Threads are the
# right tool here (not multiprocessing): each read releases the GIL while
# blocked on I/O, so this parallelizes real wall-clock time without any
# pickling/process-startup overhead. 32 is a reasonable default for local
# SSD/spinning-disk queue depths; bump it if profiling shows headroom.
DEFAULT_IO_WORKERS = 32


@dataclass
class KeyframeRecord:
    """One row = one keyframe, fully joined across all source files."""
    video_id: str
    n: int                       # 1-indexed keyframe number (CSV / object-json filename)
    npy_row: int                 # 0-indexed row into the video's .npy array
    pts_time: float              # timestamp in seconds
    frame_idx: int               # THE value submitted to the competition, not n
    fps: float
    clip_vector: np.ndarray      # shape (512,), float32
    raw_objects: dict            # untouched contents of the per-keyframe object json ({} if missing)
    image_path: str              # e.g. "keyframes/L21_V001/003.jpg", reference only


@dataclass
class VideoLoadResult:
    video_id: str
    records: list                # list[KeyframeRecord]
    video_meta: dict             # untouched media-info contents ({} if the file is absent)
    has_media_info: bool


def _object_json_path(objects_dir: Path, video_id: str, n: int) -> Path:
    """Object detection files are zero-padded to 3 digits, e.g. n=1 -> '001.json'."""
    return objects_dir / video_id / f"{n:03d}.json"


def _read_object_json(obj_path: Path, require: bool) -> dict:
    """Single-file read for one keyframe's object detections. Tries the open
    directly (one syscall) instead of exists()-then-open (two) — at ~177K
    calls across the corpus that halves the syscall count for the common
    (file present) case."""
    try:
        with open(obj_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        if require:
            raise FileNotFoundError(f"missing object json: {obj_path}") from None
        return {}


def load_video(
    video_id: str,
    dataset_root: Path,
    require_objects: bool = True,
    io_workers: int = DEFAULT_IO_WORKERS,
) -> VideoLoadResult:
    """
    Load and join all data for a single video_id.

    Raises AssertionError if the CLIP vector count and CSV row count
    disagree, or FileNotFoundError if an expected object-detection json is
    missing and require_objects=True — both real correctness checks worth
    keeping loud rather than silently producing misaligned data.

    The per-keyframe object-detection json reads (one small file per
    keyframe, ~200/video) are the dominant cost at full-corpus scale and are
    I/O-bound, not CPU-bound, so they're fanned out across io_workers threads
    (set io_workers=1 to force serial reads, e.g. for debugging).
    """
    dataset_root = Path(dataset_root)
    npy_path = dataset_root / "clip-features-32" / f"{video_id}.npy"
    csv_path = dataset_root / "map-keyframes" / f"{video_id}.csv"
    meta_path = dataset_root / "media-info" / f"{video_id}.json"
    objects_dir = dataset_root / "objects"

    vecs = np.load(npy_path).astype("float32")  # cast up from float16 for FAISS
    csv = pd.read_csv(csv_path)

    assert len(vecs) == len(csv), (
        f"[{video_id}] keyframe count mismatch: npy has {len(vecs)} rows, csv has {len(csv)} rows"
    )

    has_media_info = meta_path.exists()
    if has_media_info:
        with open(meta_path, "r", encoding="utf-8") as f:
            video_meta = json.load(f)
    else:
        video_meta = {}

    ns = [int(n) for n in csv["n"]]
    obj_paths = [_object_json_path(objects_dir, video_id, n) for n in ns]

    if io_workers > 1:
        with ThreadPoolExecutor(max_workers=io_workers) as pool:
            try:
                raw_objects_list = list(
                    pool.map(lambda p: _read_object_json(p, require_objects), obj_paths)
                )
            except FileNotFoundError as e:
                raise FileNotFoundError(f"[{video_id}] {e}") from None
    else:
        try:
            raw_objects_list = [_read_object_json(p, require_objects) for p in obj_paths]
        except FileNotFoundError as e:
            raise FileNotFoundError(f"[{video_id}] {e}") from None

    records = []
    for row_i, (_, row) in enumerate(csv.iterrows()):
        n = ns[row_i]
        npy_row = n - 1  # csv is 1-indexed, npy is 0-indexed

        records.append(
            KeyframeRecord(
                video_id=video_id,
                n=n,
                npy_row=npy_row,
                pts_time=float(row["pts_time"]),
                frame_idx=int(row["frame_idx"]),
                fps=float(row["fps"]),
                clip_vector=vecs[npy_row],
                raw_objects=raw_objects_list[row_i],
                image_path=f"keyframes/{video_id}/{n:03d}.jpg",
            )
        )

    return VideoLoadResult(
        video_id=video_id,
        records=records,
        video_meta=video_meta,
        has_media_info=has_media_info,
    )


def discover_video_ids(dataset_root: Path) -> list:
    """All video_ids present on disk, derived from clip-features-32/*.npy filenames."""
    npy_folder = Path(dataset_root) / "clip-features-32"
    return sorted(p.stem for p in npy_folder.glob("*.npy"))


def seconds_to_nearest_record(records: list, timestamp_sec: float):
    """Given a timestamp (e.g. an ASR segment start) within ONE video's
    records, find the nearest keyframe by pts_time. Caller is responsible
    for passing records already restricted to a single video_id."""
    if not records:
        raise ValueError("seconds_to_nearest_record: empty records list")
    return min(records, key=lambda r: abs(r.pts_time - timestamp_sec))
