# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Multi-signal text-to-keyframe retrieval over the AIC video corpus (873
videos / ~177k keyframes). Seven searchable signals, each embedded/file-based
(FAISS flat indices) except the fuzzy-text legs which use a local
Elasticsearch:

| Signal  | Legs | Notes |
|---|---|---|
| Keyframe | SigLIP2, CLIP ViT-B/32, RRF | frame embeddings |
| ASR | SigLIP2-ASR, Elasticsearch fuzzy, RRF | transcript segments mapped to nearest keyframe |
| Caption | SigLIP2-caption, Elasticsearch fuzzy, RRF | one row per keyframe |
| OCR | Elasticsearch fuzzy only | single leg by design, no embedding leg, no RRF |
| Summary | SigLIP2-summary, Elasticsearch fuzzy, RRF | video-level: one result per video, "group by" groups by collection (lot) instead of by video |
| Mixed | weighted RRF across Keyframe/ASR/Caption/OCR | per-signal on/off leg toggles + adjustable weights, dialog-driven |
| Hierarchy | SigLIP2 only | 3 steps: (1) SigLIP2 frame search grouped by video, (2) per-video seed-frame picker (defaults to that group's top-1 frame), (3) drill-down using the chosen seed as a new picture query scoped to that video, pulling in results up to Top-G frames/video (default 5, "Expand" bumps one video's own G by +10) |

The query box (outside TRAKE) also accepts a **picture query**: paste an
image into it and it's embedded with each leg's SigLIP2 *image* tower
instead of its text tower, then searched the same way. Picture queries are
SigLIP2-only — CLIP ViT-B/32 and every Elasticsearch fuzzy leg have no
image counterpart and are skipped (OCR, being fuzzy-only, is unavailable
for a picture query entirely), and intra-signal RRF is skipped too since a
picture query only ever has one active leg. The picture-query paragraph above is from `ui/app.py`; the backend equivalent
lives in `backend/models.py` (`is_image_query`, `siglip2_query_vec`).

A **CodaBench submission batch tool** also lives in `submission/` (entry:
`run_submission.py`): it runs a folder of `query-*-kis/qa/trake.txt` files
through the in-process Mixed-mode pipeline and writes the per-query CSV
submission files. See `submission/README.md` for the expected query types and
output format.

Everything runs as **one** Streamlit process (`streamlit run ui/app.py`).
Do not resurrect the old per-layer app scripts as separate processes — text
towers (SigLIP2, Multilingual-CLIP, ~4GB) are loaded once via
`st.cache_resource` (keyed process-wide) and shared across all signals;
three separate processes previously exhausted the Windows page file.

## Commands

```
streamlit run ui/app.py
```

Elasticsearch (required for every fuzzy leg — ASR, Caption, OCR, Summary;
degrades to an empty result with an on-page warning if unreachable):

```
docker run -d --name es -p 9200:9200 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    -e "xpack.ml.enabled=false" \
    -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
    -v es-data:/usr/share/elasticsearch/data \
    docker.elastic.co/elasticsearch/elasticsearch:8.15.0
```

The heap cap matters: ES's default heap auto-sizes to ~50% of the host's
visible RAM (up to 26 GB), which is wildly oversized for the short-text-only
indices this app builds (no vectors — those live in FAISS). Uncapped, the
container was observed using ~8.6 GB RAM for a corpus of ~300k short text
rows; capped at 512m it should stay well under 1 GB.

The `-v es-data:...` volume mount matters too: without it, the container's
`/usr/share/elasticsearch/data` is ephemeral, so every `docker rm`/recreate
loses all four fuzzy indices and the next launch has to re-bulk everything
from CSV. With the volume, indices persist across container restarts *and*
recreation, and each `ensure_*_fuzzy_index()` in `ui/app.py` checks
`es.indices.exists(...)` up front and returns immediately if the index is
already there — so re-indexing only ever happens once, the first time an
index doesn't exist yet, not on every Streamlit launch. To force a rebuild
after changing the indexed source data, delete that one index (e.g. `curl
-X DELETE localhost:9200/asr_segments`) rather than the whole container/volume.

No test suite, linter, or build step exists in this repo. `pipeline/clip_encoder.py`
has a `__main__` smoke test: `python pipeline/clip_encoder.py "some query"`.

## Layout

```
pipeline/
  config.py       paths + constants shared by the notebooks and text encoders
  clip_encoder.py Multilingual-CLIP text tower (paired with CLIP ViT-B/32 image features)
ui/
  app.py          the combined Streamlit app — all six signals, one process (~1350 lines)
index/            generated FAISS indices + CSV metadata (git-ignored), rebuilt into cache on first run
routing101.ipynb          annotated walkthrough: keyframe embeddings (SigLIP2 / CLIP ViT-B/32 / RRF)
routing101_asr.ipynb      annotated walkthrough: ASR-segment search + RRF, mapped to keyframes
routing101_caption.ipynb  annotated walkthrough: frame-caption search + RRF, mapped to keyframes
```

`ui/app.py` is a *port* of the notebooks (see each notebook for the
annotated reasoning behind a given search/fuse step) — when changing
retrieval logic, check whether the source notebook needs the matching edit
too.

## Data (external, not in this repo)

All raw data/embeddings live outside the repo, under absolute paths
hardcoded near the top of `ui/app.py` (`FRAME_SIGLIP2_GLOB`, `ASR_EMBED_DIR`,
`CAPTIONING_DIR`, `OCR_DIR`, `SUMMARY_DIR`, `MAP_KEYFRAMES_DIR`,
`THUMBNAIL_ROOT`, etc. — currently under `D:/University/Summ26/AICData*`).
Update these constants, not a config file, if the data moves.

- `AICDataExtracted/siglib_embed/*.npy` — SigLIP2 frame embeddings (768-d)
- `AICData/clip-features-32/{video_id}.npy` — CLIP ViT-B/32 frame embeddings (512-d, float16)
- `AICDataExtracted/asr_embed/*_asr_siglip768.npy` + `_frames.csv` — SigLIP2 embeddings of ASR segments (one row per segment × keyframe)
- `AICDataExtracted/transcripts/*.csv` — raw ASR segments, bulk-indexed into ES for the fuzzy leg
- `AICDataExtracted/siglip_caption/*_caption_siglip768.npy` + `_frames.csv` — SigLIP2 embeddings of frame captions (one row per keyframe)
- `AICDataExtracted/captioning/*_captions.csv` — raw frame captions, bulk-indexed into ES
- `AICDataExtracted/ocr/` — per-frame OCR text, bulk-indexed into ES (no embedding leg)
- `AICDataExtracted/summaries/` + `summary_embed/` — one-paragraph video summaries (embedded on first use, cached)
- `AICData/map-keyframes/{video_id}.csv` — per-frame timestamps, resolves ASR/text hits to a keyframe number
- `AICData/keyframes/{video_id}/{n:03d}.jpg` — thumbnails

## Architecture notes for `ui/app.py`

- **Result shape contract**: every leg/signal, once searched, is normalized
  to `{video_id, n, rank, score_label, score_val, text}` (`n` = 1-indexed
  keyframe number from `map-keyframes`) before rendering. One `render_grid()`
  and one "show more" neighbor popup (`show_neighbors`, ± `NEIGHBOR_WINDOW`
  frames by frame number) serve most signals — keep new signals conforming
  to this shape rather than adding bespoke rendering. Hierarchy is the one
  exception: its per-video drill-down groups aren't a plain ranked
  DataFrame, so it renders its own grid directly (still built from the same
  per-result dicts `df_to_results()` produces).
- **RRF fusion**: each signal has its own `rrf_fuse_*` (frame/asr/caption/
  summary) that fuses same-signal legs; `rrf_fuse_weighted` is separate —
  it fuses *across* signals for Mixed mode using per-signal weights (see
  `change_weights_dialog` / `st.session_state.mixed_weights`).
- **Index/model build is eager and up front**: the `st.status(...)` block
  near the bottom of the file builds every FAISS index and ensures every ES
  index exists *before* the sidebar renders, so the first query on any
  signal never pays a mid-search build cost. Adding a new signal means
  adding its build step there too.
- **CLIP ViT-B/32 is Keyframe-only** by current scope — ASR/Caption/Summary
  dropped their CLIP legs (SigLIP2 + fuzzy + RRF only there).
- **Elasticsearch indices are bulk-indexed lazily**, once per process, with
  an idempotent `_id` per doc (`ensure_*_fuzzy_index`) — no separate
  indexing step. If ES is unreachable, fuzzy legs degrade to an empty
  result rather than breaking other legs on the same signal.
- **`pipeline/clip_encoder.py` loading is a manual workaround**: multilingual-clip's
  own `.from_pretrained()` is incompatible with `transformers>=5`'s
  meta-device lazy loading, so the model is constructed directly
  (`MultilingualCLIP(config)`) and the fine-tuned state dict loaded on top,
  with `strict=False` to tolerate one known benign key mismatch
  (`transformer.embeddings.position_ids`) — any *other* mismatch raises.
- Scoping filters (`apply_filters` — single video or lot/collection range)
  are applied right after each leg's search, before RRF/head truncation, so
  single-leg and RRF/Mixed views respect the same filters consistently.
