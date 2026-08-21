"""run_submission.py — CLI entry for the CodaBench submission batch tool.

Runs every query*.txt file (query-*-kis.txt / query-*-qa.txt /
query-*-trake.txt) under a queries dir through the in-process Mixed-mode
pipeline and writes the per-query CodaBench CSV output.

Usage:
    python run_submission.py [--queries DIR] [--out DIR] [--round N]
                             [--top-k N] [--max-rows N] [--answer-mode MODE]
                             [--trake-signal SIG] [--frame-index n|frame_id]

Example:
    python run_submission.py --queries queries/round1 --out submissions
"""

import argparse
from pathlib import Path

from submission.config import SubmissionConfig
from submission.run_batch import run_batch


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--queries", type=Path, default=None,
                   help="dir holding query-*-kis/qa/trake.txt files (default: repo/queries)")
    p.add_argument("--out", type=Path, default=None,
                   help="dir to write the per-query .csv submission files "
                        "(default: repo/submissions)")
    p.add_argument("--round", type=int, default=0,
                   help="if >0, writes into <out>/round<N> subdir")
    p.add_argument("--top-k", type=int, default=None,
                   help="candidate pool fetched per leg before RRF fusion")
    p.add_argument("--max-rows", type=int, default=None,
                   help="row cap per query CSV (CodaBench limit is 100)")
    p.add_argument("--answer-mode", default=None, choices=["none", "caption", "ocr"],
                   help="Q&A answer hook mode (see submission/answer.py)")
    p.add_argument("--trake-signal", default=None,
                   help="signal each TRAKE event runs (default: Mixed)")
    p.add_argument("--frame-index", default=None, choices=["n", "frame_id"],
                   help="CSV Frame Idx source: n (1-based keyframe) or frame_id")

    args = p.parse_args(argv)

    cfg = SubmissionConfig()
    if args.queries:
        cfg.queries_dir = args.queries.resolve()
    if args.out:
        cfg.output_dir = args.out.resolve()
    if args.round:
        cfg.output_dir = cfg.output_dir / f"round-{args.round}"
    if args.top_k:
        cfg.top_k = args.top_k
    if args.max_rows:
        cfg.max_rows = args.max_rows
    if args.answer_mode:
        cfg.answer_mode = args.answer_mode
    if args.trake_signal:
        cfg.trake_event_signal = args.trake_signal
    if args.frame_index:
        cfg.frame_index = args.frame_index

    run_batch(cfg)


if __name__ == "__main__":
    main()