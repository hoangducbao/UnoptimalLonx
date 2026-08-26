"""
backend/od_filter.py -- OD (object-detection) text filter: fuzzy-matches
free-text, comma-separated class tokens against the offline class
vocabulary (pipeline/build_class_vocab.py -> filtered_object/class_vocab.csv)
and applies the matched classes as an AND post-filter over an
already-ranked result DataFrame, keeping only rows whose keyframe's OD
detections (filtered_object/{video_id}.csv) include every matched class.

Call sites (backend/routes/search.py) always run this AFTER a signal's own
ranking is final -- after RRF fusion for an RRF leg, after the raw score
ranking for a plain leg -- and always right before the top_k head()
truncation, so the filter sees the full fetch_k-sized candidate pool
rather than an already-truncated top_k. Per the spec this only ever
narrows results, never blanks the page: zero matched tokens is a no-op
(unmatched tokens are skipped, not blocking), and a filter that would
empty the candidate list falls back to the unfiltered ranking instead.
"""

import csv
import difflib

import pandas as pd

from . import config

# difflib.SequenceMatcher ratio threshold for token -> vocab class match.
# Loose enough to catch a plural/typo ("cars" -> "car"), tight enough to
# not conflate unrelated classes.
FUZZY_CUTOFF = 0.75

_vocab: list = None  # normalized class names, loaded once
_od_cache: dict = {}  # video_id -> {n: {normalized class_name, ...}}, loaded lazily per video


def _normalize(text) -> str:
    return " ".join(str(text).strip().lower().split())


def _load_vocab() -> list:
    global _vocab
    if _vocab is None:
        vocab = []
        if config.CLASS_VOCAB_CSV.exists():
            with open(config.CLASS_VOCAB_CSV, "r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    name = row.get("class_name")
                    if name:
                        vocab.append(name)
        _vocab = vocab
    return _vocab


def _load_od_detections(video_id: str) -> dict:
    """video_id -> {n: {normalized class_name detected in that keyframe}},
    cached process-wide (873 videos of short per-frame CSVs -- small enough
    to just accumulate as queries touch them, no eager build needed)."""
    if video_id not in _od_cache:
        path = config.FILTERED_OBJECT_DIR / f"{video_id}.csv"
        frame_classes: dict = {}
        if path.exists():
            try:
                df = pd.read_csv(path, usecols=["keyframe_id", "class_name"])
            except (ValueError, pd.errors.EmptyDataError):
                df = None
            if df is not None:
                for keyframe_id, name in zip(df["keyframe_id"], df["class_name"]):
                    if not isinstance(name, str) or not name:
                        continue
                    try:
                        n = int(keyframe_id)
                    except (TypeError, ValueError):
                        continue
                    frame_classes.setdefault(n, set()).add(_normalize(name))
        _od_cache[video_id] = frame_classes
    return _od_cache[video_id]


def match_classes(filter_text: str):
    """Free-text, comma-separated -> (matched, unmatched). `matched` is a
    deduped list of normalized vocab class names; `unmatched` is the
    trimmed original tokens that had no fuzzy hit (skipped, not blocking --
    the caller can still filter/search on whatever did match)."""
    if not filter_text or not filter_text.strip():
        return [], []
    vocab = _load_vocab()
    matched, unmatched = [], []
    seen = set()
    for token in filter_text.split(","):
        raw = token.strip()
        if not raw:
            continue
        norm = _normalize(raw)
        hit = difflib.get_close_matches(norm, vocab, n=1, cutoff=FUZZY_CUTOFF) if vocab else []
        if hit:
            if hit[0] not in seen:
                seen.add(hit[0])
                matched.append(hit[0])
        else:
            unmatched.append(raw)
    return matched, unmatched


def unmatched_warning(unmatched: list):
    if not unmatched:
        return None
    return f"OD filter: no class match for {', '.join(unmatched)} — ignored."


def apply_od_filter(df: pd.DataFrame, matched: list) -> pd.DataFrame:
    """AND post-filter: keep only rows (video_id, n) whose keyframe's OD
    detections include every class in `matched`. No-op if nothing matched
    or df lacks the needed columns; falls back to the unfiltered df if the
    filter would otherwise empty the result."""
    if df is None or df.empty or not matched:
        return df
    if "video_id" not in df.columns or "n" not in df.columns:
        return df

    def _keep(row) -> bool:
        n = row.get("n")
        if n is None or pd.isna(n):
            return False
        classes = _load_od_detections(row["video_id"]).get(int(n), set())
        return all(c in classes for c in matched)

    filtered = df[df.apply(_keep, axis=1)]
    if filtered.empty:
        return df
    return filtered.reset_index(drop=True)
