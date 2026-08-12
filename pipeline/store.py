"""
store.py — SQLite metadata store for the C1 baseline pipeline.

Modeled on old_version/pipeline/store.py's connect()/WAL/global_id pattern:
`global_id` is assigned once per keyframe, in indexing order, never reused,
and equals the row index in the FRAME FAISS index (frame_viclip768_flat_ip)
built by index_pipeline.py. That equivalence is the join point between the
frame-vector side and the metadata side.

Tables:
  videos          one row per successfully indexed video
  skipped_videos  one row per video that failed loader.py's invariant checks
  keyframes       one row per keyframe, PK = global_id. pts_time/fps/frame_idx/n
                  are nullable — a video with no aligned map-keyframes CSV is
                  still indexed, just without timestamps (tolerate partial data).
  classes         distinct object-detection class labels, class_row_id == row
                  index in the CLASS FAISS index (class_viclip768_flat_ip)
  frame_classes   join table: which classes were detected in which frame,
                  used to map object-class FAISS matches back to frames
  keyframe_text   FTS5, trigram tokenizer over a diacritic-folded shadow
                  column (`text_folded`) — OCR/caption text, empty until that
                  data exists; the pipeline must treat "empty" as a leg to
                  skip, not an error.
"""

import argparse
import sqlite3
import unicodedata
from pathlib import Path

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id        TEXT PRIMARY KEY,
    start_global_id INTEGER NOT NULL,
    num_keyframes   INTEGER NOT NULL,
    has_timestamps  INTEGER NOT NULL,   -- 0/1: map-keyframes CSV found & row-aligned
    indexed_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS skipped_videos (
    video_id TEXT PRIMARY KEY,
    reason   TEXT NOT NULL,
    at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS keyframes (
    global_id  INTEGER PRIMARY KEY,   -- == frame FAISS row index, assigned once, never reused
    video_id   TEXT NOT NULL REFERENCES videos(video_id),
    row_index  INTEGER NOT NULL,      -- 0-indexed, == npy row == filenames.csv row_index
    filename   TEXT NOT NULL,         -- e.g. "001.jpg"
    pts_time   REAL,                  -- NULL if no aligned map-keyframes row
    fps        REAL,
    frame_idx  INTEGER,               -- competition-submission frame index, if known
    n          INTEGER                -- 1-indexed keyframe number from map-keyframes, if known
);
CREATE INDEX IF NOT EXISTS idx_keyframes_video ON keyframes(video_id);
CREATE INDEX IF NOT EXISTS idx_keyframes_video_row ON keyframes(video_id, row_index);

CREATE TABLE IF NOT EXISTS classes (
    class_name   TEXT PRIMARY KEY,
    class_row_id INTEGER NOT NULL UNIQUE   -- row index in the class FAISS index
);

CREATE TABLE IF NOT EXISTS frame_classes (
    global_id  INTEGER NOT NULL REFERENCES keyframes(global_id),
    class_name TEXT NOT NULL REFERENCES classes(class_name),
    score      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_frame_classes_class ON frame_classes(class_name);
CREATE INDEX IF NOT EXISTS idx_frame_classes_global ON frame_classes(global_id);

CREATE VIRTUAL TABLE IF NOT EXISTS keyframe_text USING fts5(
    text,               -- original text, display only
    text_folded,        -- diacritic-folded + lowercased shadow column, indexed
    source UNINDEXED,   -- 'ocr' | 'caption' (neither exists yet)
    video_id UNINDEXED,
    global_id UNINDEXED,
    tokenize = 'trigram'
);
"""


def fold_diacritics(text: str) -> str:
    """Lowercase + strip diacritics so Vietnamese text folds to a base-Latin
    form for trigram matching, e.g. "Hà Nội" -> "ha noi". NFD decomposition
    alone doesn't cover đ/Đ (a distinct letter, not d + combining stroke), so
    that's folded explicitly first. Used identically at ingest time (to
    populate text_folded) and at query time (to fold the incoming query)."""
    text = text.lower().replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def connect(db_path: Path) -> sqlite3.Connection:
    """Open (creating if needed) the metadata DB, ensure schema exists, enable WAL.

    check_same_thread=False: Streamlit's cached-resource connection (ui/app.py)
    gets reused across script reruns that may land on different worker
    threads. Safe here since this pipeline never writes concurrently from
    multiple threads -- index_pipeline.py is single-threaded, and the UI is
    read-only."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Writes — callers commit; index_pipeline.py commits once per video, at the
# same boundary both FAISS indices get written to, so stores can't drift.
# ---------------------------------------------------------------------------

def insert_video(conn: sqlite3.Connection, video) -> None:
    conn.execute(
        "INSERT INTO videos (video_id, start_global_id, num_keyframes, has_timestamps, indexed_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (video.video_id, video.start_global_id, video.num_keyframes,
         int(video.has_timestamps), video.indexed_at),
    )


def insert_skipped_video(conn: sqlite3.Connection, video_id: str, reason: str, at: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO skipped_videos (video_id, reason, at) VALUES (?, ?, ?)",
        (video_id, reason, at),
    )


def insert_keyframes(conn: sqlite3.Connection, keyframes: list) -> None:
    conn.executemany(
        "INSERT INTO keyframes (global_id, video_id, row_index, filename, pts_time, fps, frame_idx, n) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [(kf.global_id, kf.video_id, kf.row_index, kf.filename, kf.pts_time, kf.fps,
          kf.frame_idx, kf.n) for kf in keyframes],
    )


def insert_classes(conn: sqlite3.Connection, class_names: list) -> None:
    """class_names: ordered list, index == row in the class FAISS index.
    INSERT OR IGNORE so re-running against an unchanged class list is a no-op."""
    conn.executemany(
        "INSERT OR IGNORE INTO classes (class_name, class_row_id) VALUES (?, ?)",
        [(name, i) for i, name in enumerate(class_names)],
    )


def insert_frame_classes(conn: sqlite3.Connection, rows: list) -> None:
    """rows: list of (global_id, class_name, score) tuples."""
    conn.executemany(
        "INSERT INTO frame_classes (global_id, class_name, score) VALUES (?, ?, ?)", rows,
    )


def insert_keyframe_text(conn: sqlite3.Connection, rows: list) -> None:
    """rows: list of (text, source, video_id, global_id) tuples. Skips empty text."""
    conn.executemany(
        "INSERT INTO keyframe_text (text, text_folded, source, video_id, global_id) "
        "VALUES (?, ?, ?, ?, ?)",
        [(t, fold_diacritics(t), s, v, g) for t, s, v, g in rows if t],
    )


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def get_known_video_ids(conn: sqlite3.Connection) -> set:
    rows = conn.execute("SELECT video_id FROM videos").fetchall()
    rows += conn.execute("SELECT video_id FROM skipped_videos").fetchall()
    return {r["video_id"] for r in rows}


def next_global_id(conn: sqlite3.Connection) -> int:
    """Next unused global_id == current row count == where the frame FAISS
    index's ntotal should also be, if the two stores are in sync."""
    row = conn.execute("SELECT COUNT(*) AS n FROM keyframes").fetchone()
    return row["n"]


def get_video(conn: sqlite3.Connection, video_id: str):
    return conn.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,)).fetchone()


def get_keyframes_by_video(conn: sqlite3.Connection, video_id: str) -> list:
    return conn.execute(
        "SELECT * FROM keyframes WHERE video_id = ? ORDER BY row_index", (video_id,)
    ).fetchall()


def get_keyframes_by_global_ids(conn: sqlite3.Connection, global_ids: list) -> dict:
    """Returns {global_id: sqlite3.Row}."""
    if not global_ids:
        return {}
    placeholders = ",".join("?" * len(global_ids))
    rows = conn.execute(
        f"SELECT * FROM keyframes WHERE global_id IN ({placeholders})", global_ids,
    ).fetchall()
    return {r["global_id"]: r for r in rows}


def get_classes_ordered(conn: sqlite3.Connection) -> list:
    """Class names ordered by class_row_id -- must match the class FAISS
    index's row order for class_row_id <-> FAISS row lookups to line up."""
    rows = conn.execute("SELECT class_name FROM classes ORDER BY class_row_id").fetchall()
    return [r["class_name"] for r in rows]


def frames_for_classes(conn: sqlite3.Connection, class_names: list) -> list:
    """rows for every (frame, class) pair among the given class_names."""
    if not class_names:
        return []
    placeholders = ",".join("?" * len(class_names))
    return conn.execute(
        f"SELECT global_id, class_name, score FROM frame_classes "
        f"WHERE class_name IN ({placeholders})", class_names,
    ).fetchall()


def search_text(conn: sqlite3.Connection, query: str, video_id: str | None = None, limit: int = 200) -> list:
    """FTS5 MATCH against the diacritic-folded shadow column, best (lowest
    bm25()) first. Optionally restricted to a single video_id (e.g. the
    UI's "search in this video only"). Caller is expected to have already
    checked the table isn't empty (this pipeline's text leg is skipped
    entirely, not queried, while no OCR/caption data exists)."""
    folded = fold_diacritics(query)
    if len(folded.strip()) < 3:
        return []  # trigram tokenizer needs >= 3 chars to match anything
    if video_id:
        rows = conn.execute(
            "SELECT global_id, video_id, text, source, bm25(keyframe_text) AS score "
            "FROM keyframe_text WHERE keyframe_text MATCH ? AND video_id = ? ORDER BY score LIMIT ?",
            (f"text_folded:{folded}", video_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT global_id, video_id, text, source, bm25(keyframe_text) AS score "
            "FROM keyframe_text WHERE keyframe_text MATCH ? ORDER BY score LIMIT ?",
            (f"text_folded:{folded}", limit),
        ).fetchall()
    return rows


def stats(conn: sqlite3.Connection) -> dict:
    def count(sql):
        return conn.execute(sql).fetchone()[0]

    return {
        "videos_indexed": count("SELECT COUNT(*) FROM videos"),
        "videos_skipped": count("SELECT COUNT(*) FROM skipped_videos"),
        "videos_without_timestamps": count("SELECT COUNT(*) FROM videos WHERE has_timestamps = 0"),
        "keyframes": count("SELECT COUNT(*) FROM keyframes"),
        "classes": count("SELECT COUNT(*) FROM classes"),
        "frame_classes_rows": count("SELECT COUNT(*) FROM frame_classes"),
        "keyframe_text_rows": count("SELECT COUNT(*) FROM keyframe_text"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the C1 baseline metadata store.")
    parser.add_argument("--db-path", type=Path, default=config.DB_PATH)
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
