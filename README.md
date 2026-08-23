# Routing101

Multi-signal text-to-keyframe retrieval over the AIC video corpus (873
videos / ~177k keyframes), plus an AIC-submission CSV export flow. Eight
searchable signals — Keyframe, ASR, Caption, OCR, Summary, Mixed, TRAKE,
Hierarchy — each embedded/file-based (FAISS flat indices) except the
fuzzy-text legs, which use a local Elasticsearch. One FastAPI process
serves both the JSON API and the static frontend.

## Layout

```
backend/
  config.py           paths + constants (data lives outside the repo, see Data below)
  main.py             FastAPI app entry -- mounts routers + static frontend/media
  common.py           shared helpers (result-shape contract, map-keyframes lookups)
  models.py           SigLIP2 text/image tower, loaded once, shared across signals
  export.py           AIC submission CSV row-generation logic (KIS/VQA/TRAKE)
  es_client.py, es_indexing.py   Elasticsearch client + bulk-indexing for the fuzzy legs
  search/             per-signal search + RRF (keyframe, asr, caption, ocr, summary, mixed, hierarchy, trake)
  routes/             FastAPI endpoints on top of search/ (+ export, playback, neighbors, query_image)
frontend/
  index.html
  js/                 app.js (signal switcher) + api.js, state.js, render.js, dialogs.js, export-dialog.js
  js/signals/          one module per signal, same render/search shape
  css/style.css
pipeline/
  config.py           paths + constants shared by the text encoders
  clip_encoder.py      Multilingual-CLIP text tower (paired with CLIP ViT-B/32 image features)
index/                 generated FAISS indices + CSV metadata (git-ignored), rebuilt on first run
```

## Signals

| Signal | Legs | Notes |
|---|---|---|
| Keyframe | SigLIP2, CLIP ViT-B/32, RRF | frame embeddings |
| ASR | SigLIP2-ASR, Elasticsearch fuzzy, RRF | transcript segments mapped to nearest keyframe |
| Caption | SigLIP2-caption, Elasticsearch fuzzy, RRF | one row per keyframe |
| OCR | Elasticsearch fuzzy only | single leg by design, no embedding leg, no RRF |
| Summary | SigLIP2-summary, Elasticsearch fuzzy, RRF | video-level: one result per video |
| Mixed | weighted RRF across Keyframe/ASR/Caption/OCR | per-signal on/off leg toggles + adjustable weights |
| Hierarchy | SigLIP2 only | frame search grouped by video, drilled down per video |
| TRAKE | reuses the other signals, one per event | ordered multi-event search: find videos where every event's best match occurs in order |

Every leg normalizes to `{video_id, n, rank, score_label, score_val, text}`
before it reaches the frontend, so one `renderGrid()` (+ neighbor/playback
popups) serves every signal except Hierarchy and TRAKE, which render their
own grouped/multi-event shapes.

## Export

Every result card's ★ button opens an export popup that generates a
ranked, deduped CSV for one AIC query (`query-p2-<#>-<kis|qa|trake>.csv`,
no header row) — confirmed mode (a picked answer + time/similarity-based
hedge rows) or unconfirmed mode (curated/ranked candidates + hedges),
capped at 100 rows per the R@k scoring model. See `backend/export.py`'s
module docstring for the exact row-generation rules.

## Data (external, not in this repo)

All raw data/embeddings live outside the repo, under absolute paths
hardcoded in `backend/config.py` (currently `D:/University/Summ26/AICData*`).
Update those constants, not a config file, if the data moves.

- `AICDataExtracted/siglib_embed/*.npy` — SigLIP2 frame embeddings (768-d)
- `AICData/clip-features-32/{video_id}.npy` — CLIP ViT-B/32 frame embeddings (512-d, float16)
- `AICDataExtracted/transcript_embed/{video_id}.npy` + `.csv` — SigLIP2 embeddings of ASR segments (one row per segment × keyframe)
- `AICDataExtracted/transcripts/{video_id}.csv` — raw ASR segments, bulk-indexed into ES for the fuzzy leg
- `AICDataExtracted/caption_embed/{video_id}.npy` + `.csv` — SigLIP2 embeddings of frame captions (one row per keyframe)
- `AICDataExtracted/captions/{video_id}.csv` — raw frame captions, bulk-indexed into ES
- `AICDataExtracted/ocr/{video_id}.csv` — per-frame OCR text, bulk-indexed into ES (no embedding leg)
- `AICDataExtracted/summaries/` + `summary_embed/` — one-paragraph video summaries (embedded on first use, cached)
- `AICData/map-keyframes/{video_id}.csv` — per-frame timestamps + native `frame_idx`, resolves ASR/text hits to a keyframe number and backs the CSV export's `n -> frame_idx` translation
- `AICData/keyframes/{video_id}/{n:03d}.jpg` — thumbnails
- `AICData/video/{video_id}.mp4` — source video, used by the playback dialogs

## Run

```
uvicorn backend.main:app --reload
```

Then open `http://localhost:8000/app/`. Elasticsearch is required for
every fuzzy leg (ASR, Caption, OCR, Summary) — degrades to an empty result
with an on-page warning if unreachable:

```
docker run -d --name es -p 9200:9200 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    -e "xpack.ml.enabled=false" \
    -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
    -v es-data:/usr/share/elasticsearch/data \
    docker.elastic.co/elasticsearch/elasticsearch:8.15.0
```

The `-v es-data:...` volume matters: without it, `docker rm`/recreate
loses all four fuzzy indices and the next launch re-bulks everything from
CSV. With the volume, `ensure_*_fuzzy_index()` in `backend/es_indexing.py`
checks `es.indices.exists(...)` up front and only (re)indexes an index
that doesn't exist yet — to force a rebuild after changing source data,
delete that one index (e.g. `curl -X DELETE localhost:9200/caption_frames`)
rather than the whole container/volume.

No test suite, linter, or build step exists in this repo. See `CLAUDE.md`
for architecture notes and conventions (result-shape contract, RRF fusion,
index build order, etc.).
