"""
store.py — SQLite metadata store for the AICPrep retrieval pipeline.

Replaces the old `unified_metadata.json` blob (loaded whole into memory,
not queryable, no incremental-update story). Everything here is keyed by
`global_id`, an integer assigned once per keyframe, in indexing order,
never reused — `global_id == i` is also FAISS row `i` in the index built
by index_pipeline.py. That equivalence is the join point between the
vector side and the metadata side; nothing in this module knows about
FAISS itself.

One SQLite file holds:
  - videos            one row per successfully indexed video
  - skipped_videos     one row per video that failed loader.py's invariant
                        checks (npy/csv mismatch, etc.) — kept so re-runs
                        don't retry the same broken video every time
  - keyframes          one row per keyframe, PK = global_id
  - keyframe_text      FTS5 virtual table, unified text-search channel
                        (object labels now; OCR/ASR appended in later phases)
"""

import argparse
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id        TEXT PRIMARY KEY,
    start_global_id INTEGER NOT NULL,
    num_keyframes   INTEGER NOT NULL,
    title           TEXT,
    author          TEXT,
    watch_url       TEXT,
    description     TEXT,
    publish_date    TEXT,
    length          INTEGER,
    has_media_info  INTEGER NOT NULL,   -- 0/1: media-info/{video_id}.json was present
    status          TEXT NOT NULL,      -- 'indexed'
    indexed_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS skipped_videos (
    video_id TEXT PRIMARY KEY,
    reason   TEXT NOT NULL,
    at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS keyframes (
    global_id  INTEGER PRIMARY KEY,   -- == FAISS row index, assigned once, never reused
    video_id   TEXT NOT NULL REFERENCES videos(video_id),
    n          INTEGER NOT NULL,      -- 1-indexed keyframe number (CSV/object-json filename)
    pts_time   REAL NOT NULL,
    frame_idx  INTEGER NOT NULL,      -- THE value submitted to the competition, not n
    image_path TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_keyframes_video ON keyframes(video_id);
CREATE INDEX IF NOT EXISTS idx_keyframes_video_pts ON keyframes(video_id, pts_time);

CREATE VIRTUAL TABLE IF NOT EXISTS keyframe_text USING fts5(
    text,
    source UNINDEXED,      -- 'objects' | 'ocr' | 'asr'
    video_id UNINDEXED,
    global_id UNINDEXED
);

CREATE TABLE IF NOT EXISTS asr_segments (
    video_id   TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time   REAL NOT NULL,
    text       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_asr_video_time ON asr_segments(video_id, start_time);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    """Open (creating if needed) the metadata DB, ensure schema exists, enable WAL."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Writes — callers are expected to commit (index_pipeline.py commits once per
# video, at the same boundary the FAISS index gets appended to, so the two
# stores never drift relative to each other).
# ---------------------------------------------------------------------------

def insert_video(conn: sqlite3.Connection, video) -> None:
    """video: index_pipeline.IndexedVideo (or any object with matching attrs)."""
    conn.execute(
        "INSERT INTO videos (video_id, start_global_id, num_keyframes, title, author, "
        "watch_url, description, publish_date, length, has_media_info, status, indexed_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'indexed', ?)",
        (
            video.video_id, video.start_global_id, video.num_keyframes,
            video.title, video.author, video.watch_url, video.description,
            video.publish_date, video.length, int(video.has_media_info),
            video.indexed_at,
        ),
    )


def insert_skipped_video(conn: sqlite3.Connection, video_id: str, reason: str, at: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO skipped_videos (video_id, reason, at) VALUES (?, ?, ?)",
        (video_id, reason, at),
    )


def insert_keyframes(conn: sqlite3.Connection, keyframes: list) -> None:
    """keyframes: list of index_pipeline.IndexedKeyframe (or matching-attr objects)."""
    conn.executemany(
        "INSERT INTO keyframes (global_id, video_id, n, pts_time, frame_idx, image_path) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (kf.global_id, kf.video_id, kf.n, kf.pts_time, kf.frame_idx, kf.image_path)
            for kf in keyframes
        ],
    )


def insert_keyframe_text(conn: sqlite3.Connection, rows: list) -> None:
    """rows: list of (text, source, video_id, global_id) tuples. Skips empty text."""
    conn.executemany(
        "INSERT INTO keyframe_text (text, source, video_id, global_id) VALUES (?, ?, ?, ?)",
        [r for r in rows if r[0]],
    )


def insert_asr_segments(conn: sqlite3.Connection, video_id: str, segments: list) -> None:
    """segments: list of (start_time, end_time, text) tuples."""
    conn.executemany(
        "INSERT INTO asr_segments (video_id, start_time, end_time, text) VALUES (?, ?, ?, ?)",
        [(video_id, s, e, t) for s, e, t in segments],
    )


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def get_known_video_ids(conn: sqlite3.Connection) -> set:
    """Every video_id already accounted for — indexed OR skipped. Used by
    index_pipeline.py to diff against on-disk video_ids and find only the
    genuinely new ones (the incremental-add mechanism)."""
    rows = conn.execute("SELECT video_id FROM videos").fetchall()
    rows += conn.execute("SELECT video_id FROM skipped_videos").fetchall()
    return {r["video_id"] for r in rows}


def next_global_id(conn: sqlite3.Connection) -> int:
    """Next unused global_id == current row count == where FAISS index.ntotal
    should also be, if the two stores are in sync."""
    row = conn.execute("SELECT COUNT(*) AS n FROM keyframes").fetchone()
    return row["n"]


def get_video(conn: sqlite3.Connection, video_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,)).fetchone()


def get_keyframes_by_video(conn: sqlite3.Connection, video_id: str) -> list:
    return conn.execute(
        "SELECT * FROM keyframes WHERE video_id = ? ORDER BY n", (video_id,)
    ).fetchall()


def get_keyframes_by_global_ids(conn: sqlite3.Connection, global_ids: list) -> dict:
    """Returns {global_id: sqlite3.Row}, joined with the owning video's title
    (LEFT JOIN so a missing media-info row doesn't drop the keyframe)."""
    if not global_ids:
        return {}
    placeholders = ",".join("?" * len(global_ids))
    rows = conn.execute(
        f"SELECT k.*, v.title, v.watch_url, v.author FROM keyframes k "
        f"LEFT JOIN videos v ON v.video_id = k.video_id "
        f"WHERE k.global_id IN ({placeholders})",
        global_ids,
    ).fetchall()
    return {r["global_id"]: r for r in rows}


def search_text(conn: sqlite3.Connection, query: str, video_ids: list | None = None, limit: int = 200) -> list:
    """FTS5 MATCH search over keyframe_text, best (lowest bm25()) first.
    Optionally restricted to a set of video_ids (e.g. CLIP's candidate set)."""
    if video_ids:
        placeholders = ",".join("?" * len(video_ids))
        sql = (
            f"SELECT global_id, video_id, text, source, bm25(keyframe_text) AS score "
            f"FROM keyframe_text WHERE keyframe_text MATCH ? AND video_id IN ({placeholders}) "
            f"ORDER BY score LIMIT ?"
        )
        params = [query, *video_ids, limit]
    else:
        sql = (
            "SELECT global_id, video_id, text, source, bm25(keyframe_text) AS score "
            "FROM keyframe_text WHERE keyframe_text MATCH ? ORDER BY score LIMIT ?"
        )
        params = [query, limit]
    return conn.execute(sql, params).fetchall()


# ---------------------------------------------------------------------------
# Stats — replaces check.py's row-count sanity check
# ---------------------------------------------------------------------------

def stats(conn: sqlite3.Connection) -> dict:
    def count(sql):
        return conn.execute(sql).fetchone()[0]

    return {
        "videos_indexed": count("SELECT COUNT(*) FROM videos"),
        "videos_skipped": count("SELECT COUNT(*) FROM skipped_videos"),
        "videos_missing_media_info": count("SELECT COUNT(*) FROM videos WHERE has_media_info = 0"),
        "keyframes": count("SELECT COUNT(*) FROM keyframes"),
        "keyframe_text_rows": count("SELECT COUNT(*) FROM keyframe_text"),
        "asr_segments": count("SELECT COUNT(*) FROM asr_segments"),
    }


if __name__ == "__main__":
    HERE = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Inspect the AICPrep metadata store.")
    parser.add_argument("--db-path", type=Path, default=HERE.parent / "index" / "aic_metadata.db")
    parser.add_argument("--stats", action="store_true", help="Print row-count summary")
    parser.add_argument("--skipped", action="store_true", help="List skipped videos + reasons")
    args = parser.parse_args()

    conn = connect(args.db_path)
    if args.skipped:
        for row in conn.execute("SELECT video_id, reason, at FROM skipped_videos ORDER BY at"):
            print(f"{row['video_id']}: {row['reason']} ({row['at']})")
    else:
        for key, value in stats(conn).items():
            print(f"{key:28s} {value:,}")
    conn.close()
