# submission/ — CodaBench batch submission tool

Runs a folder of organizer **query `.txt`** files through this repo's
in-process retrieval pipeline (**Mixed mode**) and writes the per-query
**CodaBench `.csv`** submission files.

## Query file → output mapping

| Input file suffix | Query type        | CSV line format                          |
|---|---|---|
| `query-*-kis.txt`    | Textual Known Item Search | `<video>, <Frame Idx>`                 |
| `query-*-qa.txt`     | Question & Answer         | `<video>, <Frame Idx>, <Answer>`      |
| `query-*-trake.txt`  | TRAKE (timed event chain) | `<video>, <id1>, <id2>, …, <idN>`     |

`<video>` = the video file stem (e.g. `L00_V000`). `<Frame Idx>` = the
pipeline's keyframe number `n` (`frame_id` `+ 1`).

## Run

```
python run_submission.py --queries queries/round1 --out submissions
```

- Looks for `*.txt` under `--queries` (default `repo/queries`).
- Writes `<query-name>.csv` under `--out` (default `repo/submissions`),
  optionally into a `round-N` subdir with `--round N`.
- Warns and skips un-parsable/MISSING files; each query gets its own CSV.

## Behavior notes

- **In-process**: the batch imports the FastAPI `backend.search` modules
  directly and warms them up once, so the ~4 GB of model weights load once —
  honoring the repo's single-process constraint. Run it from the repo root.
- **Mixed mode**: KIS/Q&A run `mixed_results()` (weighted RRF over
  Keyframe/ASR/Caption/OCR). TRAKE events default to the `Mixed` signal too.
- **Row cap**: ≤ `cfg.max_rows` (default 100) rows per query, per CodaBench.
- **Q&A answers**: plug in your VQA model in `submission/answer.py`
  `generate_answer()`. Default `answer_mode="none"` emits an empty answer for
  a format-only run; `"caption"`/`"ocr"` are best-effort ES-text fallbacks.
- **CSV rules**: UTF-8, comma-delimited, **no header row**, CRLF by default,
  quoting only where CodaBench requires it (comma / quote / newline /
  leading-trailing space) — escape `"` as `""`, cap answers at 100 chars.

## Layout

```
run_submission.py   CLI entry point
submission/
  config.py         paths + Mixed weights/legs + CSV rules (dataclass)
  query.py          query file -> typed Query (KIS/QA/TRAKE) parsing
  pipeline.py       in-process wrappers + warmup
  answer.py         pluggable Q&A answer hook
  csvout.py         CodaBench CSV writers (quoting, encoding, line endings)
  run_batch.py      orchestration
```