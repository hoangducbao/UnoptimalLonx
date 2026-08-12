"""
clean_objects.py — Step 2: turn one keyframe's raw object-detection JSON
into a small, deduplicated, confidence-filtered set of labels.

Raw detections (as provided by the organizers, Faster-R-CNN/OpenImages-style)
are noisy: confidence trails off to near-zero (~0.002 seen in real sample
data), and the same object is often detected 20+ times from overlapping
boxes. This module reduces that to something worth pushing into a text
index (keyframe_text FTS5) alongside future OCR/ASR text.
"""

from dataclasses import dataclass


@dataclass
class CleanedObjects:
    video_id: str
    n: int
    labels: list          # deduplicated, confidence-filtered, e.g. ["Skyscraper", "Tower", "Boat"]
    label_counts: dict    # how many raw detections survived filtering per label (diagnostic)
    text: str             # ready-to-index string, e.g. "Skyscraper Tower Boat"


def clean_detection_json(
    raw: dict,
    video_id: str = "",
    n: int = 0,
    min_confidence: float = 0.3,
    max_labels: int | None = None,
) -> CleanedObjects:
    """
    Filter + dedup one keyframe's raw object-detection JSON.

    min_confidence: drop detections below this score (0.3 is a reasonable
        starting point — worth tuning against a validation sample of real
        VQA/KIS queries once scored feedback exists).
    max_labels: optionally cap distinct labels kept, sorted by confidence
        descending. None keeps everything that passes the threshold.
    """
    scores = raw.get("detection_scores", [])
    entities = raw.get("detection_class_entities", [])

    assert len(scores) == len(entities), (
        f"[{video_id} n={n}] detection_scores/detection_class_entities length "
        f"mismatch: {len(scores)} vs {len(entities)}"
    )

    best_score_per_label: dict = {}
    for score_str, label in zip(scores, entities):
        score = float(score_str)
        if score < min_confidence:
            continue
        if label not in best_score_per_label or score > best_score_per_label[label]:
            best_score_per_label[label] = score

    sorted_labels = sorted(best_score_per_label.items(), key=lambda kv: kv[1], reverse=True)
    if max_labels is not None:
        sorted_labels = sorted_labels[:max_labels]

    labels = [label for label, _ in sorted_labels]
    label_counts = {label: entities.count(label) for label in labels}

    return CleanedObjects(
        video_id=video_id,
        n=n,
        labels=labels,
        label_counts=label_counts,
        text=" ".join(labels),
    )


def clean_records(
    records: list,  # list[loader.KeyframeRecord]
    min_confidence: float = 0.3,
    max_labels: int | None = None,
) -> list:
    """Clean object detections for every record from loader.load_video(...).records.
    Output is index-aligned 1:1 with `records` — safe to zip directly."""
    return [
        clean_detection_json(
            r.raw_objects, video_id=r.video_id, n=r.n,
            min_confidence=min_confidence, max_labels=max_labels,
        )
        for r in records
    ]
