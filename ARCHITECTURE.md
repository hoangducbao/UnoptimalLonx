# Architecture

How Routing101 is put together. For getting it running in the first
place, see [`README.md`](README.md) instead — this file assumes it's
already up.

## Layout

```
backend/
  config.py           paths + constants (data lives outside the repo, see README's Prerequisites)
  main.py             FastAPI app entry -- mounts routers + static frontend/media
  common.py           shared helpers (result-shape contract, map-keyframes lookups)
  models.py           SigLIP2 text/image tower, loaded once, shared across signals
  export.py           AIC submission CSV row-generation logic (KIS/VQA/TRAKE)
  od_filter.py         object-detection text filter (fuzzy class match)
  metadata_filter.py   structured per-lot metadata facet filter
  es_client.py, es_indexing.py   Elasticsearch client + bulk-indexing for the fuzzy legs
  search/             per-signal search + RRF (keyframe, asr, caption, ocr, summary, mixed, hierarchy, trake)
  routes/             FastAPI endpoints on top of search/ (+ export, facets, playback, neighbors, query_image)
frontend/
  index.html           main app shell
  export.html           standalone Export CSV page (opened in its own tab, see Export below)
  js/                 app.js (signal switcher) + api.js, state.js, render.js, dialogs.js
  js/export-dialog.js, export-page.js, export-ui.js   Export CSV tab: opener handoff, entry point, UI
  js/signals/          one module per signal, same render/search shape
  css/style.css
pipeline/
  build_class_vocab.py  builds the OD class vocabulary od_filter.py matches against
  *.csv                 per-lot metadata extracted upstream (metadata_filter.py's source)
index/                  generated FAISS indices + CSV metadata (git-ignored), rebuilt on first run
```

## Signals

| Signal | Legs | Notes |
|---|---|---|
| Keyframe | SigLIP2 only | frame embeddings; CLIP ViT-B/32 + its Multilingual-CLIP query-time text encoder (XLM-RoBERTa-large) were removed entirely -- that text tower alone cost ~4.6GB RAM lazily loaded, dwarfing every other model/index in this system combined |
| ASR | SigLIP2-ASR, Elasticsearch fuzzy, RRF | transcript segments mapped to nearest keyframe |
| Caption | SigLIP2-caption, Elasticsearch fuzzy, RRF | one row per keyframe |
| OCR | Elasticsearch fuzzy only | single leg by design, no embedding leg, no RRF |
| Summary | SigLIP2-summary, Elasticsearch fuzzy, RRF | video-level: one result per video |
| Mixed | many independent sub-queries (Keyframe/ASR/Caption/OCR), fused by weighted RRF | one query + one signal + one weight (0-3) per sub-query, add/remove freely; optional "Show transcript" attaches ASR text under every result regardless of which sub-query ranked it |
| Hierarchy | SigLIP2 only | frame search grouped by video, drilled down per video |
| TRAKE | reuses the other signals, one per event | ordered multi-event search: find videos where every event's best match occurs in order; optional context (E0) query is always matched via Summary, boosting a video's score independent of the ordering constraint |

Every leg normalizes to `{video_id, n, rank, score_label, score_val, text}`
(`backend/common.py::df_to_results`) before it reaches the frontend, so one
`renderGrid()` (+ neighbor/playback popups) serves every signal except
Hierarchy and TRAKE, which render their own grouped/multi-event shapes.

## Filters

Three independent scoping dimensions, applied to every signal/leg after
its own ranking and before the top-k cutoff: **video/collection** scope
(a single `video_id`, or a lot range like `L21-L30`, optionally excluded
instead of restricted-to), the **Object filter** (free-text class names,
fuzzy-matched against an offline OD vocabulary, e.g. "car, dog, red car"),
and the **Metadata filter** (a dropdown over structured per-lot facets like
subject/province, populated from `/api/facets`).

## Export architecture

Every result card's ★ button opens the Export CSV UI in its own **browser
tab** (`frontend/export.html`), not an in-page popup — it stays in sync
with whatever you're currently searching in the original tab, via a
same-origin `window.opener` handoff (`state.js` exposes
`window.__routing101` for this; `export-page.js` reads a live reference
to the opener's `exportState`, not a frozen snapshot, so "Similars" always
reflects the opener's most recent search). It generates a ranked, deduped
CSV for one AIC query (`query-p2-<#>-<kis|qa|trake>.csv`, no header row),
capped at 99 rows per the R@k scoring model (Final Score = average of
R@k for k in {1, 5, 20, 50, 100}, where R@k = max score among the first k
rows — only the best-scoring row within each threshold band matters, so a
duplicate of an already-placed row never helps, only wastes a slot).

- **KIS/VQA** (`backend/export.py::_generate_export_flat`) — confirmed
  mode (a picked answer + its nearest keyframes by time as hedges) or
  unconfirmed mode (a curated ordered list of answer candidates), then a
  similar-semantic tier (confirmed mode: a fresh visual search seeded by
  the confirmed frame itself, `similar_candidates_for_frame`; unconfirmed
  mode: the opener tab's own ranked results) and a filler tier
  (nearest-by-time keyframes of each similar), until the row budget is
  spent. The VQA answer text box is a plain typed field in both modes —
  no LLM auto-fill, that was a planned later phase that isn't happening;
  whatever's typed goes straight into the CSV's quoted answer column.

  A **"Keyframes" checkbox** (next to "Confirmed") switches the answer
  between an indexed keyframe (`{video_id, n}`, the above) and a raw
  **native** frame (`{video_id, frame_idx}`) straight from video
  playback, unchecked. Unchecked, the Neighbours/Similars preview grids
  are replaced by a TRAKE-style curation panel — an inline `<video>` plus
  an add button that captures whatever frame is currently playing;
  confirmed mode caps it at one frame ("Switch to this frame" swaps it,
  captioned by video_id), unconfirmed mode is a plain list of candidates
  in the order added ("Cand 1", "Cand 2", ..., no temporal sort, unlike
  TRAKE's events) — no Generate-rows/cache/merge step either way, the
  curated list(s) go straight into `/api/export`. The Video ID/Frame ID/
  Change row is dropped entirely in this mode (the curation panel's own
  video is the only way to add a frame); VQA reuses that vacated topbar
  slot for its answer text box instead, KIS just leaves it empty. Row
  generation still runs the same tiers server-side with nothing to
  preview: the confirmed/answers tier is built in frame_idx space via
  TRAKE's own `_event_neighbour_stream` neighbour logic
  (`generate_export(..., keyframes=False)`) instead of
  `nearest_keyframes_by_time`, and confirmed mode's similar-semantic
  search snaps to the nearest indexed keyframe first
  (`similar_candidates_for_native_frame`), since a raw frame has no
  SigLIP2 embedding of its own to search from. A frame captured live from
  a video-playback dialog (no keyframe `n` at all) opens KIS/VQA with
  Keyframes unchecked by default; re-checking it snaps whatever's curated
  to its nearest keyframe (`/api/export/nearest-keyframe`).
- **TRAKE** — no confirmed/unconfirmed distinction at all. A
  **curate → cache → merge** flow instead, entirely inside the Export
  tab's TRAKE panel:
  1. **Curate** one video at a time: an inline `<video>` player (load by
     typing a video id, or seeded from whatever triggered the ★) plus an
     "Add current frame as event" button that captures a canvas snapshot
     of the frame currently playing into an ordered event list (drag to
     reorder, ✕ to remove). The Frame ID box can add a raw frame number
     too. No Neighbours/Similars preview for TRAKE — an event's only
     "similar" pool is what row generation computes server-side, not
     something to browse.
  2. **Generate rows**: POSTs that video's `{video_id, frame_idxs}` to
     `/api/export/trake-rows`, which runs
     `backend/export.py::generate_trake_rows` — row 1 is the curated
     picks as-is; rows 2–99 zip each event's *k*-th nearest neighbour
     (keyframe-index distance if that event's pick is itself a keyframe,
     else plain native-frame-number distance, since a non-keyframe pick
     has no embedding to search a "similar" pool with at all), enforcing
     per-row temporal ordering (event *i*'s frame < event *j*'s for
     *i*<*j*) via random interpolation whenever a neighbour stream runs
     dry or would break that order. The ≤99 resulting sequences are
     cached client-side, keyed by `video_id` — repeatable for as many
     candidate videos as you want to compare, each just adding another
     cache entry.
  3. **Export**: check which cached videos to include and drag them into
     a priority order; the rows are interleaved client-side (no backend
     round-trip, no re-reading anything) — each checked video's own row 1
     first in priority order, then row 2/3/... round-robin in that same
     order until the 99-row cap or every video's rows are spent — and
     POSTed to `/api/export/trake-write`, which only formats + returns
     CSV text for already-resolved rows (the one file this whole flow
     ever writes to disk).

  Not limited to an actual TRAKE search: any signal's result card, a
  Neighbours/Similars preview pick (KIS/VQA), or a frame captured live
  from a video-playback dialog can seed a curation session's first event
  — and a video-playback frame is no longer TRAKE-only either: it opens
  with all three query types available (KIS/VQA default to Keyframes
  unchecked, per above), seeding both the TRAKE and native KIS/VQA
  curation panels with the same frame up front.

## Frontend caching

`frontend/` is mounted via a `NoCacheStaticFiles` subclass
(`backend/main.py`) that stamps `Cache-Control: no-cache` on every
response — forces revalidation (an unchanged file still 304s off its
ETag) rather than letting the browser reuse a stale copy under default
heuristic freshness. Added after exactly that bit — a browser silently
running an old `export-ui.js` against an already-updated backend, with no
error anywhere — cost real debugging time. `/media` (keyframes/video) is
deliberately left with default caching: those files are genuinely
immutable per `video_id`/frame, unlike frontend source that changes
underneath an already-open tab during development.

## Performance notes

- **Thread-pool tuning** (`backend/config.py::tune_thread_pools`, called
  once from `backend/main.py`'s lifespan): CPU-only torch defaults to
  num-cores intraop *and* num-cores interop threads, and FAISS's own
  OpenMP pool defaults to num-cores on top of that — left uncapped, the
  pools compound into far more live threads than the box has cores.
  `CPU_BUDGET = cpu_count - 2` caps torch's intraop pool and FAISS's OpenMP
  pool, leaving interop at 1 thread and headroom for uvicorn/the OS.
- Model/index loading is eager, at process startup, once — not lazy
  per-request, not rebuilt per-request (mirrors the single-process,
  ~4GB-of-loaded-weights constraint from this project's original
  Streamlit prototype).

## Conventions

No test suite, linter, or build step exists in this repo. Every search
leg normalizes to the same result-shape contract (see Signals above)
specifically so one render path can serve most of the frontend without
per-signal special-casing — keep new signals/legs conforming to that
shape rather than inventing a parallel one.
