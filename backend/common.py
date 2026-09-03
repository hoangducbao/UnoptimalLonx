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
    fingerprint a PIL.Image either).

    The active query-chunking strategy (models.py) is folded in because it
    changes what a long query's embedding legs return, and every leg's TTL
    cache keys on this -- without it, flipping the setting would keep serving
    the previous strategy's ranking until the entries aged out. The ES legs
    aren't affected by the strategy at all; they just pay one extra miss
    after a switch, which is cheaper than giving them a second key shape."""
    from .models import get_query_chunk_strategy

    prefix = get_query_chunk_strategy() + "|"
    if isinstance(query, Image.Image):
        return prefix + "img:" + hashlib.sha1(query.tobytes()).hexdigest()
    return prefix + "txt:" + query


def faiss_search_pooled(index, qmat: np.ndarray, k: int, per_vec_k: int = None):
    """Search `index` with one or more query vectors; return (ids, scores)
    for the top-k rows, at most one entry per row.

    One query vector -- an image query, a short text query, or any query
    under truncate/mean_chunks -- is FAISS's own result handed straight back,
    scores and all, so those paths rank exactly as they did before chunking
    existed.

    Several vectors (chunks_separate, one per chunk of a long query) each get
    their own ranked list, and the lists are fused with **RRF**, the same
    reciprocal-rank fusion every signal already uses to combine its legs:
    a row's score is the sum of 1/(RRF_K + rank) over the chunks that
    retrieved it. That rewards a row for placing well against *several*
    chunks, which is what makes a long query behave like a conjunction of its
    clauses -- a max-pool would instead let one strongly-matching clause
    carry a row that ignored the rest of the query.

    Two consequences of fusing rather than pooling:

    * The returned scores are RRF scores (order 1/RRF_K, so ~0.016 and down),
      not cosine similarities. Only their order is meaningful, and only
      relative to each other -- so a long query's leg shows a different scale
      of number than a short one's. Everything downstream fuses on `rank`
      rather than `score`, so nothing but the displayed figure changes.
    * Ties are common (one appearance at the same rank in different chunks
      scores identically), so they're broken by best cosine, then by id --
      deterministic, and it prefers the row that actually matched harder.

    `per_vec_k` sets how deep each chunk's own list goes, separately from the
    `k` distinct rows returned, for the caller that overfetches for its own
    reasons (search/summary.py fetches extra chunk rows so its per-video
    max-pool has enough to work with).
    """
    qmat = l2_normalize(np.asarray(qmat, dtype="float32").reshape(-1, index.d))
    n = min(per_vec_k or k, index.ntotal)
    scores, ids = index.search(qmat, n)
    if qmat.shape[0] == 1:
        return ids[0], scores[0]
    fused, best = {}, {}
    for row_ids, row_scores in zip(ids, scores):
        rank = 0
        for gid, score in zip(row_ids, row_scores):
            gid = int(gid)
            if gid == -1:
                continue
            # Rank counts only the hits actually returned, so a short list
            # padded with -1 doesn't shift the ranks after the gap.
            rank += 1
            fused[gid] = fused.get(gid, 0.0) + 1.0 / (config.RRF_K + rank)
            score = float(score)
            if score > best.get(gid, float("-inf")):
                best[gid] = score
    out_ids = sorted(fused, key=lambda g: (-fused[g], -best[g], g))[:k]
    return (np.asarray(out_ids, dtype="int64"),
            np.asarray([fused[g] for g in out_ids], dtype="float32"))

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


def parse_lot_range(text: str, exclude: bool = False):
    """'L21-L30' / 'P01-P10' / '21-30' / 'L21' / 'P01' -> (lo, hi, exclude) lot range, or None if
    blank/unparsable. `exclude` flips apply_filters from "keep only this
    range" (the default) to "drop this range" -- the sidebar's "Exclude"
    checkbox next to "Search in collection"."""
    text = (text or "").strip().upper()
    if not text:
        return None
    m = re.match(r"^(?:L|P|ADL_?)?(\d+)\s*-\s*(?:L|P|ADL_?)?(\d+)$", text)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        lo, hi = (lo, hi) if lo <= hi else (hi, lo)
        return (lo, hi, exclude)
    m = re.match(r"^(?:L|P|ADL_?)?(\d+)$", text)
    if m:
        n = int(m.group(1))
        return (n, n, exclude)
    return None


def video_lot_num(video_id: str):
    m = re.match(r"^(?:L|P|ADL_?)?(\d+)", str(video_id).upper())
    return int(m.group(1)) if m else None


def video_lot_str(video_id: str) -> str:
    lot = video_lot_num(video_id)
    if lot is not None:
        prefix = "P" if str(video_id).upper().startswith("P") else "L"
        return f"{prefix}{lot}"
    return str(video_id)


def apply_filters(df: pd.DataFrame, video_filter: str, lot_range) -> pd.DataFrame:
    """Restrict a leg's result df to a single video_id and/or a lot range
    (or, when that range's exclude flag is set, drop it instead of keeping
    only it), applied right after search (before RRF/head truncation) so
    both single-leg and RRF views respect the same filters."""
    if df is None or df.empty:
        return df
    out = df
    video_filter = (video_filter or "").strip().upper()
    if video_filter:
        out = out[out["video_id"].astype(str).str.upper() == video_filter]
    if lot_range:
        lo, hi, exclude = lot_range
        lots = out["video_id"].map(video_lot_num)
        in_range = lots.notna() & (lots >= lo) & (lots <= hi)
        out = out[~in_range] if exclude else out[in_range]
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


_video_path_cache: dict = {}
_all_video_files_index: dict = {}
_has_indexed_kaggle_videos: bool = False


def _build_kaggle_video_index():
    global _has_indexed_kaggle_videos
    if _has_indexed_kaggle_videos:
        return
    _has_indexed_kaggle_videos = True
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return
    # Scan /kaggle/input for all video files (mp4, mkv, webm, avi)
    exts = {".mp4", ".mkv", ".webm", ".avi"}
    for root, _, files in os.walk(input_root):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in exts:
                _all_video_files_index[p.stem.lower()] = p


def find_video_path(video_id: str) -> Path | None:
    """Finds the actual video file path for a given video_id across flat directories,
    nested lot directories (Videos_L23, L23, video/), or Kaggle input mounts (e.g. degarr)."""
    if not video_id:
        return None
    vid_clean = str(video_id).strip()
    vid_key = vid_clean.lower()
    if vid_key in _video_path_cache:
        return _video_path_cache[vid_key]

    exts = [".mp4", ".mkv", ".webm", ".avi"]

    # 1. Direct in config.VIDEO_DIR
    if config.VIDEO_DIR and config.VIDEO_DIR.exists():
        for ext in exts:
            p = config.VIDEO_DIR / f"{vid_clean}{ext}"
            if p.exists():
                _video_path_cache[vid_key] = p
                return p

        # Check lot subfolders within config.VIDEO_DIR
        lot_str = video_lot_str(vid_clean)
        candidate_subdirs = [
            config.VIDEO_DIR / lot_str,
            config.VIDEO_DIR / f"Videos_{lot_str}",
            config.VIDEO_DIR / f"Videos_{lot_str}" / "video",
            config.VIDEO_DIR / "video" / lot_str,
            config.VIDEO_DIR / "video",
            config.VIDEO_DIR / "videos",
        ]
        for sdir in candidate_subdirs:
            if sdir.exists():
                for ext in exts:
                    p = sdir / f"{vid_clean}{ext}"
                    if p.exists():
                        _video_path_cache[vid_key] = p
                        return p

    # 2. Check Kaggle /kaggle/input (datasets like degarr, aic2026-dataset, etc.)
    if config.IS_KAGGLE or Path("/kaggle/input").exists():
        _build_kaggle_video_index()
        if vid_key in _all_video_files_index and _all_video_files_index[vid_key].exists():
            res = _all_video_files_index[vid_key]
            _video_path_cache[vid_key] = res
            return res

    _video_path_cache[vid_key] = None
    return None


def video_url(video_id: str) -> str:
    return f"/api/playback/stream/{video_id}"


_map_keyframes_cache: dict = {}


def load_map_keyframes(video_id: str):
    if video_id not in _map_keyframes_cache:
        path = config.MAP_KEYFRAMES_DIR / f"{video_id}.csv"
        if path.exists():
            df = pd.read_csv(path)
            # Defends against stray junk on the header row (seen on one file
            # in the wild: a leading backtick turned "n" into "`n", which
            # made every mk["n"] lookup below raise KeyError and 500 the
            # whole request) -- normalize instead of trusting the header
            # byte-for-byte, so a similarly mangled file degrades to working
            # rather than crashing.
            df.columns = [str(c).strip().strip("`") for c in df.columns]
            _map_keyframes_cache[video_id] = df
        else:
            _map_keyframes_cache[video_id] = None
    return _map_keyframes_cache[video_id]


def get_video_keyframes_meta(video_id: str) -> list:
    """Returns a list of keyframe descriptors for video_id for frontend scrubber/fallback."""
    mk = load_map_keyframes(video_id)
    if mk is None or mk.empty:
        return []
    out = []
    for _, r in mk.iterrows():
        try:
            n_val = int(r["n"])
            pts_val = float(r.get("pts_time", 0.0))
            fps_val = float(r.get("fps", 25.0))
            frame_idx_val = int(r.get("frame_idx", n_val))
            out.append({
                "n": n_val,
                "pts_time": pts_val,
                "fps": fps_val,
                "frame_idx": frame_idx_val,
                "thumbnail_url": thumbnail_url(video_id, n_val),
            })
        except Exception:
            continue
    return out


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


def frame_idx_for_n(video_id: str, n):
    """n (1-indexed keyframe ordinal, the field every result carries) ->
    frame_idx (raw video frame index) -- used by backend/export.py, since
    AIC submission CSVs expect frame_idx, not n. Returns None if
    unresolvable."""
    mk = load_map_keyframes(video_id)
    if mk is None or mk.empty or n is None or pd.isna(n):
        return None
    hit = mk.loc[mk["n"] == int(n)]
    if hit.empty:
        return None
    return int(hit.iloc[0]["frame_idx"])


def valid_ns_for_video(video_id: str) -> set:
    """Every n that actually exists for video_id -- used by backend/export.py
    to filter out-of-range hedge offsets (n-1, n+2, ...) before they'd
    resolve to a nonexistent keyframe."""
    mk = load_map_keyframes(video_id)
    if mk is None or mk.empty:
        return set()
    return set(mk["n"].astype(int))


def n_for_frame_idx(video_id: str, frame_idx: int):
    """Reverse of frame_idx_for_n: native frame_idx -> keyframe n, only if
    frame_idx exactly matches an existing keyframe's frame_idx. Used by
    TRAKE row generation (backend/export.py::generate_trake_rows) to tell
    apart a keyframe-backed pick (neighbours ranked by keyframe-index
    distance) from a raw native-frame pick with no embedding behind it
    (neighbours ranked by plain frame-number distance instead). Returns
    None if frame_idx isn't any keyframe's, including when map-keyframes
    itself is missing for video_id."""
    mk = load_map_keyframes(video_id)
    if mk is None or mk.empty or frame_idx is None:
        return None
    hit = mk.loc[mk["frame_idx"] == int(frame_idx)]
    if hit.empty:
        return None
    return int(hit.iloc[0]["n"])


def nearest_keyframe_n_for_frame_idx(video_id: str, frame_idx: int):
    """Nearest keyframe n to a native frame_idx by |frame_idx - keyframe's
    own frame_idx| -- unlike n_for_frame_idx, always returns *some*
    keyframe (as long as the video has any indexed at all), not just an
    exact match. Used to seed a visual-similarity search, or a "Keyframes"
    checkbox re-check, from a raw native frame (Export tab, Keyframes
    unchecked) that has no embedding/n of its own. Returns None only if
    map-keyframes has no rows for video_id."""
    mk = load_map_keyframes(video_id)
    if mk is None or mk.empty or frame_idx is None:
        return None
    idx = (mk["frame_idx"] - int(frame_idx)).abs().idxmin()
    return int(mk.loc[idx, "n"])


def native_frame_range_for_video(video_id: str):
    """(lo, hi) bound on real frame_idx values for video_id, used by TRAKE
    row generation to keep interpolated/hedge frame numbers in range. Only
    as precise as map-keyframes' own frame_idx column (the true last video
    frame can run a little past the last keyframe's) -- good enough for a
    filler-row bound, not claimed to be the exact video length. Falls back
    to a generous synthetic range if map-keyframes is missing entirely, so
    callers never have to special-case "no data" themselves."""
    mk = load_map_keyframes(video_id)
    if mk is None or mk.empty:
        return (0, 10**7)
    return (0, int(mk["frame_idx"].max()))


def video_fps_for_video(video_id: str) -> float:
    """Any one row's fps for video_id -- map-keyframes stores the same fps
    on every row for a given video. Used to start TRAKE curation playback
    from a bare video_id, before any keyframe/timestamp is known yet.
    Falls back to the same 25.0 default backend/routes/playback.py already
    uses when a specific frame's fps can't be resolved."""
    mk = load_map_keyframes(video_id)
    if mk is None or mk.empty:
        return 25.0
    return float(mk.iloc[0]["fps"])


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
