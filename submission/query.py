"""submission/query.py — read organizer query*.txt files into typed Query
objects, one per file.

File-name convention (from the competition brief):

    <name>-kis.txt    -> Textual Known Item Search
    <name>-qa.txt     -> Question & Answer
    <name>-trake.txt  -> Temporal Retrieval and Alignment of Key Events

KIS and Q&A files hold one free-text query. A TRAKE file holds an ordered
list of event sub-queries (one per expected Frame Idx on the row).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class QueryType(str, Enum):
    KIS = "kis"
    QA = "qa"
    TRAKE = "trake"

    @staticmethod
    def from_filename(name: str) -> "QueryType | None":
        """Infer the query kind from the file stem's suffix (kis/qa/trake)."""
        stem = Path(name).stem.lower()
        for t in QueryType:
            if stem.endswith(f"-{t.value}") or stem.endswith(f"_{t.value}"):
                return t
        return None


@dataclass
class Query:
    name: str          # file stem, e.g. "query-1-kis"
    type: QueryType
    text: str          # KIS/QA: the query text; TRAKE: the raw event block
    events: list = field(default_factory=list)  # TRAKE only: ordered event texts

    @property
    def output_name(self) -> str:
        """Coda expects one .csv per query, named after the query file."""
        return f"{self.name}.csv"


# ---------------------------------------------------------------------------
# TRAKE event parsing
# ---------------------------------------------------------------------------

def _strip_numbering(line: str) -> str:
    """'1. foo' / '1) foo' / '(1) foo' -> 'foo'"""
    m = re.match(r"^\s*(?:\(\s*\d+\s*\)|\d+\s*[.)])\s*", line)
    return line[m.end():] if m else line


def parse_trake(text: str) -> list:
    """Split (a) TRAKE query text into an ordered list of event sub-queries.

    Expected layout (BTC's typical 'one event per line'):
        1. A person walks into a room
        2. The person sits down
        3. ...
    Also handles a single line with numbered events ("(1) A (2) B").
    """
    raw = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not raw:
        return []

    # Single line: try numbered-bullet split, else treat the whole line.
    if len(raw) == 1:
        line = raw[0]
        chunks = re.split(r"\s+(?:\(\d+\)|\d+[.)])\s*", line)
        if len(chunks) > 1:
            return [_strip_numbering(c).strip() for c in chunks if c.strip()]
        return [line]

    # Multi-line: one event per line.
    return [_strip_numbering(ln) for ln in raw]


# ---------------------------------------------------------------------------
# File -> typed Query
# ---------------------------------------------------------------------------

def parse_query_file(path: Path) -> Query:
    name = path.stem
    qtype = QueryType.from_filename(name)
    if qtype is None:
        raise ValueError(
            f"cannot infer query type from {name!r} — expected a name ending "
            "in -kis, -qa, or -trake (e.g. query-1-kis.txt)."
        )

    text = path.read_text(encoding="utf-8-sig").strip()
    q = Query(name=name, type=qtype, text=text)

    if qtype is QueryType.TRAKE:
        events = parse_trake(text)
        if not events:
            raise ValueError(f"{name}: TRAKE has no parsed events from {text!r}")
        q.events = events
    elif not text:
        raise ValueError(f"{name}: empty query text.")

    return q