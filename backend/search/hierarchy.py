"""
backend/search/hierarchy.py -- Hierarchy Search, three steps:
  1. A SigLIP2 frame search (text or picture query), grouped by video like
     Keyframe's "group by video".
  2. Per video, a seed-frame picker -- which of that group's own frames
     becomes the NEW picture query for step 3. Defaults to the group's
     top-1 frame; changing it only affects that one video.
  3. Drill-down: the chosen seed frame is embedded and searched scoped to
     that one video, pulling in up to Top-G frames total per video
     (default G=5, "Expand" bumps one video's own G by +10).
This only ever uses the SigLIP2 frame leg -- fuzzy legs have no
picture-query counterpart, and the drill-down step is a picture query by
construction (the seed is a frame's own thumbnail), so there's no
meaningful text/RRF path to offer here at all. Ported from
ui/app.py:1132-1179 (hierarchy_expand_group).
"""

import pandas as pd
from PIL import Image

from .. import config
from .. import metadata_filter as md
from ..common import apply_filters, df_to_results, thumbnail_disk_path
from . import keyframe as kf


def hierarchy_expand_group(video_id: str, frames: list, top_g: int, fetch_k: int, seed_n: int = None) -> list:
    """`frames` is one video's results from the base search (Step 1),
    already in rank order (best first). If it's already at/over Top-G, just
    truncate. Otherwise, embed a seed frame's thumbnail as a new SigLIP2
    picture query, search scoped to this video only, and append
    not-yet-present frames (in that scoped search's own rank order) until
    Top-G is hit or the scoped search runs out of candidates.

    `seed_n` (Step 2): which frame number to use as that query -- defaults
    to the group's own top-1 frame when not given, but the caller can pass
    any frame number a user picked instead, applying only to this one
    video's drill-down."""
    if len(frames) >= top_g:
        return frames[:top_g]

    seed_n = seed_n if seed_n is not None else frames[0]["n"]
    thumb = thumbnail_disk_path(video_id, seed_n)
    if not thumb.exists():
        return frames  # no seed image on disk -- nothing to drill down with

    try:
        seed_image = Image.open(thumb).convert("RGB")
    except Exception:
        return frames

    have_ns = {f["n"] for f in frames}
    needed = top_g - len(frames)
    # A plain video-id filter (not apply_filters' lot-range path) -- the
    # scoped search still runs over the whole FAISS index first, so k needs
    # enough headroom that this one video's frames actually surface in it.
    scoped_df = apply_filters(kf.search_siglip2_frame(seed_image, k=max(fetch_k, top_g * 40)), video_id, None)
    if scoped_df is None or scoped_df.empty:
        return frames

    extra_rows = []
    for _, row in scoped_df.sort_values("rank").iterrows():
        if len(extra_rows) >= needed:
            break
        n = int(row["n"])
        if n in have_ns:
            continue
        extra_rows.append(row)
        have_ns.add(n)

    if not extra_rows:
        return frames
    extra_results = df_to_results(pd.DataFrame(extra_rows), "score")
    return frames + extra_results


def base_search_grouped(query, fetch_k, video_filter, lot_filter, top_k, facet_field="", facet_value=""):
    """Step 1: SigLIP2 frame search, grouped by video_id in first-occurrence
    (= best rank) order -- same grouping render_grid's group_mode="video"
    does, hand-rolled here since Hierarchy needs the per-group list for
    steps 2-3, not just a flat rendered grid (ui/app.py:2114-2130).

    The metadata facet filter (subject/province) is applied here, at Step 1,
    same tier as video_filter/lot_filter -- video-level scoping, unlike
    od_filter's per-frame content filter, which Hierarchy doesn't use at
    all. Step 3's drill-down (hierarchy_expand_group) doesn't need it
    re-applied: it's already scoped to one video_id that passed this
    filter, so every frame it finds necessarily belongs to that same
    already-matching video."""
    base_df = apply_filters(kf.search_siglip2_frame(query, k=fetch_k), video_filter, lot_filter)
    base_df = md.apply_facet_filter(base_df, facet_field, facet_value)
    if base_df is None or base_df.empty:
        return []
    base_results = df_to_results(base_df.head(top_k), "score")
    groups: dict = {}
    order = []
    for r in base_results:
        vid = r["video_id"]
        if vid not in groups:
            groups[vid] = []
            order.append(vid)
        groups[vid].append(r)
    return [{"video_id": vid, "frames": groups[vid]} for vid in order]
