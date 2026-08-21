# Routing101

Multi-layer text-to-keyframe retrieval over the AIC video corpus (873
videos / ~177k keyframes): keyframe embeddings, ASR-transcript embeddings,
and frame-caption embeddings, each searchable standalone or fused with
Reciprocal Rank Fusion (RRF). One combined Streamlit UI over all three
layers. Everything is embedded/file-based (FAISS flat indices, on-disk
caches) except the fuzzy-text legs, which use a local Elasticsearch.

## Layout

```
pipeline/
  config.py          paths + constants shared by the notebooks and the text encoders
  clip_encoder.py     Multilingual-CLIP text tower, paired with CLIP ViT-B/32 image features
ui/
  app.py              combined Streamlit UI — all three layers below, one process
backend/              FastAPI rewrite of ui/app.py (see backend/main.py)
frontend/             static HTML/CSS/JS frontend served by the FastAPI app
index/                generated FAISS indices (git-ignored), reused across app runs
routing101.ipynb        annotated walkthrough: keyframe embeddings (SigLIP2 / CLIP ViT-B/32 / RRF)
routing101_asr.ipynb     annotated walkthrough: ASR-segment search + RRF, mapped to keyframes
routing101_caption.ipynb annotated walkthrough: frame-caption search + RRF, mapped to keyframes
run_submission.py      CLI entry: run a folder of query*.txt through the pipeline -> CodaBench CSV
submission/            CodaBench batch tool (config.py, query.py, pipeline.py, answer.py, csvout.py, run_batch.py)
queries/               place organizer query*.txt files here (see queries/examples/)
```

## Layers

| Layer | Legs kept | Text encoder(s) |
|---|---|---|
| Keyframe | SigLIP2, CLIP ViT-B/32, RRF | SigLIP2 text tower, Multilingual-CLIP |
| ASR | SigLIP2-ASR, Elasticsearch fuzzy, RRF | SigLIP2 text tower |
| Caption | SigLIP2-caption, Elasticsearch fuzzy, RRF | SigLIP2 text tower |

The SigLIP2 and Multilingual-CLIP text towers are loaded once per process
(`st.cache_resource`, keyed process-wide) and shared across all three
layers — `ui/app.py` is meant to run as a single `streamlit run` process;
running the old per-layer apps as separate processes each duplicated the
~4GB of model weights and was the direct cause of paging-file exhaustion.

CLIP ViT-B/32 is Keyframe-only by current scope — the ASR/Caption layers
dropped their CLIP legs (SigLIP2 + fuzzy + RRF only).

## Data

- `AICDataExtracted/siglib_embed/*.npy` — SigLIP2 frame embeddings (768-d)
- `AICData/clip-features-32/{video_id}.npy` — CLIP ViT-B/32 frame embeddings
  (512-d, float16 on disk)
- `AICDataExtracted/asr_embed/*_asr_siglip768.npy` + `_frames.csv` —
  SigLIP2 embeddings of ASR segments, one row per (segment × keyframe)
- `AICDataExtracted/transcripts/*.csv` — raw ASR transcript segments, bulk
  -indexed into Elasticsearch for the fuzzy leg
- `AICDataExtracted/siglip_caption/*_caption_siglip768.npy` + `_frames.csv`
  — SigLIP2 embeddings of frame captions (one row per keyframe)
- `AICDataExtracted/captioning/*_captions.csv` — raw frame captions, bulk
  -indexed into Elasticsearch for the fuzzy leg
- `AICData/map-keyframes/{video_id}.csv` — per-frame timestamps, used to
  resolve the ASR fuzzy leg's segment hits to a keyframe number
- `AICData/keyframes/{video_id}/{n:03d}.jpg` — thumbnails

## UI

```
streamlit run ui/app.py
```

Segmented control picks the layer (Keyframe / ASR / Caption); each layer
has one checkbox per leg + RRF. Hover a result to see a 4x zoom preview.
Click **Show more** on any result to open a popup with the ±7 neighboring
frames (by frame number) from the same video.

The fuzzy legs bulk-index their source CSVs into Elasticsearch on first
use per process (cached, idempotent `_id` per doc) — no separate indexing
step needed. Requires a local ES reachable at `http://localhost:9200`:

```
docker run -d --name es -p 9200:9200 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    -e "xpack.ml.enabled=false" \
    -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
    docker.elastic.co/elasticsearch/elasticsearch:8.15.0
```

The heap cap and disabled ML module keep ES's footprint small — this
app's ES indices only hold short text rows for fuzzy matching (no vector
data, that's all in FAISS), so ES's default auto-sized heap (up to ~50%
of host RAM) is far more than needed.

If ES isn't reachable, the fuzzy leg degrades to an empty result (with an
on-page warning) rather than breaking the other legs.

## Submission tool (CodaBench batch)

This repo ships an automatic **CodaBench submission batch tool** that reads the
competition (BTC) query files (`query-1-kis.txt`, `query-2-qa.txt`,
`query-3-trake.txt`, …), runs each through the in-process **Mixed-mode**
pipeline, and writes the per-query `.csv` submission files in the exact format
BTC expects.

### Requirements

- `pip install -r requirements.txt` (torch, faiss, transformers, fastapi,
  uvicorn, elasticsearch, cachetools, …).
- The AIC data + embeddings must be at the hardcoded paths in
  `backend/config.py` (currently `D:/University/Summ26/AICData*`). Without it
  the tool still runs but every query yields an **empty** CSV, with warnings.
- Elasticsearch (`http://localhost:9200`) is optional — without it the fuzzy
  legs warn and simply contribute nothing.
- The tool runs **in-process** (one set of model weights), so run it from the
  repo root — never alongside a separate UI/backend process that already holds
  the ~4 GB of weights.

### Step 1 — put the query files in a folder

Drop the BTC query `.txt` files anywhere (e.g. `queries/round1/`). The file
**name suffix decides the query type**:

| File suffix | Query type | CSV line format |
|---|---|---|
| `query-*-kis.txt` | Textual Known Item Search | `<video>, <Frame Idx>` |
| `query-*-qa.txt` | Question & Answer | `<video>, <Frame Idx>, <Answer>` |
| `query-*-trake.txt` | TRAKE (ordered event chain) | `<video>, <Frame ID_1>, ..., <Frame ID_N>` |

- KIS / Q&A files: one free-text query per file.
- TRAKE files: **one event per line** (numbering `1.` / `1)` / `(1)` is
  optional and stripped automatically). The output row will carry exactly
  `N` frame ids in chronological order.
- Ready-made examples live in `queries/examples/`.

### Step 2 — run the batch

```
python run_submission.py --queries queries/round1 --out submissions --round 1
```

| Flag | Effect | Default |
|---|---|---|
| `--queries DIR` | folder holding the `query-*.txt` files | `queries/` |
| `--out DIR` | where the `.csv` files are written | `submissions/` |
| `--round N` | write into `<out>/round-N` (nice for keeping round runs apart) | no subdir |
| `--top-k N` | candidates fetched per leg before RRF fusion | `100` |
| `--max-rows N` | max rows per query CSV (CodaBench cap is 100) | `100` |
| `--answer-mode none\|caption\|ocr` | how Q&A answers are filled (see below) | `none` |
| `--trake-signal SIG` | signal used for every TRAKE event | `Mixed` |
| `--frame-index n\|frame_id` | which frame numbering to write | `n` (1-based keyframe) |

For each query file you get `<name>.csv` (e.g. `query-1-kis.csv`).

### Step 3 — the output CSV follows the CodaBench rules

- Pure plain-text `.csv`: **UTF-8**, comma-delimited, **no header row**,
  CRLF line endings by default.
- At most **100 rows** per query (each row = one predicted result, ranked best
  first).
- `<Frame Idx>` is written from the pipeline's keyframe number `n`
  (`--frame-index frame_id` switches to the 0-based index instead).
- Q&A `<Answer>`: quoted only when the answer contains a comma, a double quote,
  a newline, or leading/trailing whitespace. Quotes inside are escaped `""`
  (e.g. `... ,"Anh ấy nói ""Xin chào"""`). Answers are capped at 100 chars.
- **Answers are a plug-in**: the retrieval pipeline has no free-text VQA head.
  With the default `--answer-mode none` every answer is left empty (good for a
  dry run of the format). To actually answer, implement
  `submission/answer.py -> generate_answer()` and, optionally, use
  `--answer-mode caption` / `--answer-mode ocr` for a best-effort answer from
  the matched frame's caption/OCR text (needs Elasticsearch).

See `submission/README.md` for the module layout and more details.
