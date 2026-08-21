"""submission/run_batch.py — run every query*.txt file in a folder through the
in-process pipeline and write the CodaBench CSV output for each query.

Flow per file:
  1. classify by suffix (kis / qa / trake)
  2. run Mixed-mode search (or TRAKE over Mixed events)
  3. normalize to CodaBench rows
  4. write a UTF-8, comma-delimited, header-less .csv with correct quoting
"""

from __future__ import annotations

from pathlib import Path

from . import csvout, pipeline
from .answer import generate_answer
from .config import SubmissionConfig
from .query import Query, QueryType, parse_query_file


def run_batch(cfg: SubmissionConfig) -> list:
    qdir = Path(cfg.queries_dir)
    if not qdir.is_dir():
        raise SystemExit(f"queries dir not found: {qdir}")

    qfiles = sorted(qdir.glob("*.txt"))
    if not qfiles:
        print(f"[run_batch] no *.txt query files under {qdir}")

    print("[run_batch] warming up in-process pipeline (one set of model weights)...")
    pipeline.warmup(cfg)

    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = []
    for qf in qfiles:
        q = parse_query_file(qf)
        out_path = outdir / q.output_name

        if q.type is QueryType.KIS:
            results = pipeline.mixed_results(q.text, cfg)
            csvout.write_kis(out_path, results, cfg)

        elif q.type is QueryType.QA:
            cands = pipeline.mixed_results(q.text, cfg)
            rows = [
                {"video_id": c["video_id"], "n": c["n"],
                 "answer": generate_answer(q.text, cfg, c)}
                for c in cands
            ]
            csvout.write_qa(out_path, rows, cfg)

        elif q.type is QueryType.TRAKE:
            results = pipeline.trake_results(q.events, cfg)
            csvout.write_trake(out_path, results, cfg)

        else:
            raise RuntimeError(f"unhandled query type: {q.type}")

        summary.append((q.name, len(results), out_path))
        print(f"[run_batch] {q.name} -> {out_path} ({len(results)} rows)")

    print("[run_batch] done.")
    return summary