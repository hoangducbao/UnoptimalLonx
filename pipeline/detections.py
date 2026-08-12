"""
detections.py — Step 2: load filtered_detections.csv once, expose the
distinct class-label vocabulary (for the class-embedding index) and a
per-video, per-frame-number class -> score lookup (for the frame_classes
join table).

Columns: lesson, video, frame_file, class_mid, class_name, class_label_id,
score, box_0, box_1, box_2, box_3. `frame_file` is "NNN.json" -- the same
zero-padded frame number as the embeddings' "NNN.jpg" filenames, just a
different extension. Not every keyframe has a detection row.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

import config

_FRAME_NUM_RE = re.compile(r"(\d+)\.\w+$")


def _frame_number(frame_file: str):
    m = _FRAME_NUM_RE.search(frame_file)
    return int(m.group(1)) if m else None


@dataclass
class DetectionStore:
    by_video: dict = field(default_factory=dict)  # video_id -> {frame_number: {class_name: score}}
    classes: list = field(default_factory=list)     # sorted distinct class_name

    def frame_classes(self, video_id: str, frame_number: int) -> dict:
        return self.by_video.get(video_id, {}).get(frame_number, {})


def load_detections(csv_path: Path = config.DETECTIONS_CSV) -> DetectionStore:
    """Loaded once, fully in memory (~466k rows -- cheap for a dev scaffold)."""
    if not csv_path.exists():
        return DetectionStore()

    df = pd.read_csv(csv_path, usecols=["video", "frame_file", "class_name", "score"])
    df["frame_number"] = df["frame_file"].map(_frame_number)
    df = df.dropna(subset=["frame_number"])
    df["frame_number"] = df["frame_number"].astype(int)

    by_video = defaultdict(lambda: defaultdict(dict))
    for video_id, frame_number, class_name, score in df[
        ["video", "frame_number", "class_name", "score"]
    ].itertuples(index=False):
        frame_map = by_video[video_id][frame_number]
        if class_name not in frame_map or score > frame_map[class_name]:
            frame_map[class_name] = float(score)

    classes = sorted(df["class_name"].unique().tolist())
    return DetectionStore(
        by_video={v: dict(frames) for v, frames in by_video.items()},
        classes=classes,
    )
