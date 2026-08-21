"""submission/run_batch.py — run every query*.txt file in a folder through the
in-process pipeline and write the CodaBench CSV output for each query.

Flow per file:
  1. classify by suffix (kis / qa / trake)
  2. run Mixed-mode search (or TRAKE over Mixed events)
  3. normalize to CodaBench rows
  4. write a UTF-8, comma-delimited, header-less .csv with correct quoting

Robustness: one un-parsable/failing query file is reported and skipped — it
never aborts the rest of the batch.
"""

from __future__ import annotations

from pathlib import Path

from . import csvout, pipeline
from .answer import generate_answer
from .config import SubmissionConfig
from .query import QueryType, parse_query_file


def run_batch(cfg: SubmissionConfig) -> list:
    qdir = Path(cfg.queries_dir)
    if not qdir.is_dir():
        raise SystemExit(f"queries dir not found: {qdir}")

    qfiles = sorted(qdir.glob("*.txt"))
    if not qfiles:
        print(f"[run_batch] no *.txt query files under {qdir}")

    print("[run_batch] warming up in-process pipeline (one set of model weights)...")
    pipeline.warmup(cfg)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    summary, errors = [], []
    for qf in qfiles:
        try:
            q = parse_query_file(qf)
            out_path = Path(cfg.output_dir) / q.output_name

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

        except Exception as e:  # noqa: BLE001 — keep the batch alive on a single bad file
            errors.append((qf.name, str(e)))
            print(f"[run_batch] SKIPPED {qf.name}: {e}")

    for name, err in errors:
        print(f"[run_batch] failed: {name} ({err})")
    print(f"[run_batch] done — {len(summary)} ok, {len(errors)} skipped.")
    return summary