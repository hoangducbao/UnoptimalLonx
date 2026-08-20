"""
backend/common.py -- shared helpers used by every search/* module. Ported
from ui/app.py:170-276, 1315-1331 (video_id_from_filename through
image_b64, plus df_to_results). One behavioral change from the Streamlit
app: thumbnails/video are served as URLs via FastAPI's StaticFiles mount
(backend/main.py) instead of base64 data URIs, so `thumbnail_path` becomes
`thumbnail_url` and there's no `image_b64` equivalent needed here -- see
the rewrite plan's Decisions section 1.
"""

import hashlib
import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from PIL import Image

from . import config


def query_hash(query) -> str:
    """Cache-key fragment for a query -- a text query is already hashable
    as-is, but a picture query (PIL.Image) isn't, so hash its raw pixel
    bytes instead (mirrors ui/app.py's _QUERY_HASH_FUNCS, which existed for
    the same reason: st.cache_data's default hasher doesn't know how to
    fingerprint a PIL.Image either)."""
    if isinstance(query, Image.Image):
        return "img:" + hashlib.sha1(query.tobytes()).hexdigest()
    return "txt:" + query

# ---------------------------------------------------------------------------
# Filename / id parsing
# ---------------------------------------------------------------------------

def video_id_from_filename(path_str: str, suffixes: tuple) -> str:
    stem = Path(path_str).stem
    for suffix in suffixes:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    mat = mat.astype("float32", copy=True)
    faiss.normalize_L2(mat)
    return mat


def parse_lot_range(text: str):
    """'L21-L30' / 'L21' / '21-30' -> (lo, hi) lot numbers, or None if blank/unparsable."""
    text = (text or "").strip().upper()
    if not text:
        return None
    m = re.match(r"^L?(\d+)\s*-\s*L?(\d+)$", text)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return (lo, hi) if lo <= hi else (hi, lo)
    m = re.match(r"^L?(\d+)$", text)
    if m:
        return int(m.group(1)), int(m.group(1))
    return None


def video_lot_num(video_id: str):
    m = re.match(r"^L(\d+)", str(video_id).upper())
    return int(m.group(1)) if m else None


def video_lot_str(video_id: str) -> str:
    lot = video_lot_num(video_id)
    return f"L{lot}" if lot is not None else str(video_id)


def apply_filters(df: pd.DataFrame, video_filter: str, lot_range) -> pd.DataFrame:
    """Restrict a leg's result df to a single video_id and/or a lot range,
    applied right after search (before RRF/head truncation) so both single-leg
    and RRF views respect the same filters."""
    if df is None or df.empty:
        return df
    out = df
    video_filter = (video_filter or "").strip().upper()
    if video_filter:
        out = out[out["video_id"].astype(str).str.upper() == video_filter]
    if lot_range:
        lo, hi = lot_range
        lots = out["video_id"].map(video_lot_num)
        out = out[lots.notna() & (lots >= lo) & (lots <= hi)]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Thumbnails / video / map-keyframes
# ---------------------------------------------------------------------------

def thumbnail_url(video_id: str, n) -> str:
    if n is None or pd.isna(n):
        return ""
    return f"/media/keyframes/{video_id}/{int(n):03d}.jpg"


def thumbnail_disk_path(video_id: str, n) -> Path:
    return config.THUMBNAIL_ROOT / video_id / f"{int(n):03d}.jpg"


def video_url(video_id: str) -> str:
    return f"/media/video/{video_id}.mp4"


_map_keyframes_cache: dict = {}


def load_map_keyframes(video_id: str):
    if video_id not in _map_keyframes_cache:
        path = config.MAP_KEYFRAMES_DIR / f"{video_id}.csv"
        _map_keyframes_cache[video_id] = pd.read_csv(path) if path.exists() else None
    return _map_keyframes_cache[video_id]


def nearest_keyframe_n_by_time(video_id: str, t: float):
    """Nearest map-keyframes row (by pts_time) to timestamp t -- used for
    the ASR fuzzy leg, which is segment-level only (no direct frame_id)."""
    mk = load_map_keyframes(video_id)
    if mk is None or mk.empty or pd.isna(t):
        return None
    idx = (mk["pts_time"] - t).abs().idxmin()
    return int(mk.loc[idx, "n"])


def keyframe_timestamp(video_id: str, n):
    """Symmetric counterpart to nearest_keyframe_n_by_time: direct n ->
    (pts_time, fps) lookup, used by TRAKE to place a matched frame on the
    video timeline. Returns (None, None) if unresolvable."""
    mk = load_map_keyframes(video_id)
    if mk is None or mk.empty or n is None or pd.isna(n):
        return None, None
    hit = mk.loc[mk["n"] == int(n)]
    if hit.empty:
        return None, None
    row = hit.iloc[0]
    return float(row["pts_time"]), float(row["fps"])


# ---------------------------------------------------------------------------
# Result-shape contract -- every leg/signal normalizes to this dict shape
# before it's returned to the frontend (mirrors ui/app.py's df_to_results).
# ---------------------------------------------------------------------------

def df_to_results(df: pd.DataFrame, score_col: str, text_col: str = None) -> list:
    if df is None or df.empty:
        return []
    out = []
    for _, r in df.iterrows():
        n = r.get("n")
        if n is None or pd.isna(n):
            continue
        n = int(n)
        text = r.get(text_col) if text_col else None
        out.append({
            "video_id": r["video_id"], "n": n, "rank": int(r["rank"]),
            "score_label": score_col, "score_val": float(r[score_col]),
            "text": text if isinstance(text, str) else None,
            "thumbnail_url": thumbnail_url(r["video_id"], n),
        })
    return out
