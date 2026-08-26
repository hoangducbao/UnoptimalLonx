"""
backend/metadata_filter.py -- structured facet filter over the per-lot
metadata extracted into pipeline/*.csv (subject/topic for L25, province for
L27-29, ...). Not every extracted CSV yields a good *facet* -- a facet needs
low cardinality (a handful of pickable values), so this only exposes the two
dimensions that qualify: "subject" (L25, 9 values) and "province" (L27-29,
low single digits). L26's dish_name, and L27-29's site/host, are too
high-cardinality to browse as a dropdown -- they stay in the CSVs for
reference/fuzzy-search use, not wired in here.

Same call shape as backend/od_filter.py: an AND post-filter applied after a
leg's own ranking, right alongside apply_filters (video/lot scope) in
backend/routes/search.py, since this is exactly that -- a third scope
dimension, just video-level metadata instead of video_id/lot-number. Unlike
od_filter, an empty result here is left empty rather than falling back to
unfiltered: od_filter's fuzzy per-frame class match can plausibly miss
everything in a small candidate pool even when the class truly is present
elsewhere, but this filter is an exact match against a hand-verified value
picked from a dropdown -- "no video in this pool is tagged X" is a real,
correct answer, not a filter that's too aggressive to trust.
"""

import csv

import pandas as pd

from . import config

# field -> {video_id: {value, ...}} -- a set per video since one field
# (province) can hold more than one value for a single video (L28's
# multi-province finale row).
_facets: dict = None


def _add(table: dict, video_id: str, field: str, value: str) -> None:
    value = (value or "").strip()
    if not value:
        return
    table.setdefault(field, {}).setdefault(video_id, set()).add(value)


def _load() -> dict:
    global _facets
    if _facets is not None:
        return _facets
    table: dict = {}

    subj_path = config.PIPELINE_DIR / "L25_subjects_topics.csv"
    if subj_path.exists():
        with open(subj_path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                _add(table, row["video_id"], "subject", row.get("subject"))

    for name in ("L27_episodes.csv", "L28_episodes.csv", "L29_episodes.csv"):
        path = config.PIPELINE_DIR / name
        if not path.exists():
            continue
        with open(path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                # L28's finale row holds "An Giang / Cần Thơ / Hậu Giang" --
                # split so the video is findable under each province, not
                # just as one long compound string nobody would pick.
                for province in (row.get("province") or "").split("/"):
                    _add(table, row["video_id"], "province", province)

    _facets = table
    return _facets


def get_facets() -> dict:
    """{field: [sorted distinct values]} for the frontend's facet dropdown."""
    table = _load()
    return {field: sorted({v for values in per_video.values() for v in values})
            for field, per_video in table.items()}


def apply_facet_filter(df: pd.DataFrame, field: str, value: str) -> pd.DataFrame:
    """AND post-filter: keep only rows whose video_id is tagged `value`
    under `field`. No-op if field/value blank, field unknown, or df lacks
    video_id."""
    field = (field or "").strip()
    value = (value or "").strip()
    if df is None or df.empty or not field or not value:
        return df
    table = _load()
    per_video = table.get(field)
    if per_video is None or "video_id" not in df.columns:
        return df
    mask = df["video_id"].map(lambda vid: value in per_video.get(vid, ()))
    return df[mask].reset_index(drop=True)
