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
                        (768 profile at index/routing101_*, 1152 and 1536 at index/<dim>/routing101_* -- see Embedding profiles)
```

## Embedding profiles

Three SigLIP2 checkpoints, and therefore three vector dimensions, three sets
of precomputed `.npy` files, and three FAISS index trees. Picked once from the
`R101_EMBED` environment variable at process start (`backend/config.py`'s
`_PROFILES`), never from inside the app:

| | `768` (default) | `1152` | `1536` |
|---|---|---|---|
| Checkpoint | `siglip2-base-patch16-384` | `siglip2-so400m-patch14-384` | `siglip2-giant-opt-patch16-384` |
| Frames | `siglib_embed/` | `1152embed/1152keyframe/` | `1536embed/1536keyframe/` |
| ASR | `transcript_embed/` | `1152embed/1152transcript/` | `1536embed/1536transcript/` |
| Caption | `caption_embed/` | `1152embed/1152caption/` | `1536embed/1536caption/` |
| Summary | `summary_embed/` | `1152embed/1152summary/` | `1536embed/1536summary/` |
| FAISS | `index/routing101_*` | `index/1152/routing101_*` | `index/1536/routing101_*` |
| Resident | ~2.9 GB | ~5.5 GB | ~9.4 GB |
| Launch | `run_768.bat` → `:8000` | `run_1152.bat` → `:8001` | `run_1536.bat` → `:8002` |

One process per profile on its own port (`run_768.bat` / `run_1152.bat` /
`run_1536.bat`, thin wrappers over the shared `_run_common.bat` bootstrap;
non-Windows sets `R101_EMBED` directly), so two can run at once and answer
the same query in two tabs; the header pill (`/api/profile` →
`frontend/js/app.js`) says which one a tab is talking to, since they are
otherwise identical on screen. **Not** a runtime switch: at ~5.5GB for 1152
and ~8GB for 1536, holding several in one process would multiply a footprint
this system has already trimmed once on purpose (see the Keyframe row in
Signals below, on the removed M-CLIP text tower). Measured resident is the
working set; 1536's private commit runs ~2x that (~20GB), so on a 32GB box
pair 1536 with 768 rather than with 1152, and never run all three.

Everything that isn't an embedding is shared and never duplicated per
dimension: the four Elasticsearch indices (verified: the 1152 and 1536
transcript segment ids match `transcripts/*.csv` exactly, so every profile's
embedding legs key on the same `segment_id` space the fuzzy legs do), `map-keyframes`,
thumbnails, video, the OD vocabulary, the metadata facets, and the whole
export flow.

Two profile-shaped differences in the data itself, both handled in the
search modules rather than by reshaping the files:

- **ASR** — the 1152 and 1536 transcript CSVs are segment-only, with no
  `frame_id` column. `build_siglip_asr_index()` falls back to
  `nearest_keyframe_n_by_time()`, the same resolution the ES fuzzy leg has
  always used, applied once at build time. Coverage differs by profile: 1152
  is missing ASR embeddings around L25 and reaches 773 of the corpus's 873
  videos, while 1536 was embedded after that gap was filled.
- **Summary** — the 1152 and 1536 summaries are embedded chunk-by-chunk
  (`chunks_separate`: 2501 chunks over 785 videos) rather than one vector
  per summary, because SigLIP2's text tower only sees 64 tokens (see
  **Long queries** below) and a summary runs well past that — the 768
  profile silently truncated most of every summary it embedded. So the index holds one row per chunk, and
  `search_siglip_summary()` overfetches and keeps each video's best-scoring
  chunk (a max-pool) to hand one row per video to `rrf_fuse_summary`, which
  keys on `video_id` alone. A hit's text is the chunk that scored, not the
  whole paragraph — so a fused card can show chunk text from the SigLIP2
  leg or the full summary from the fuzzy leg, whichever ranked the video
  first.

## Long queries

SigLIP2's text tower has a hard 64-token context — `Siglip2TextModel`'s
`position_embedding` is `nn.Embedding(64, hidden)`, so there is no length to
raise and no RoPE to extrapolate. Anything longer used to be truncated at
encode time, with a warning as the only trace.

`backend/models.py` now splits an over-window query instead. `chunk_text()`
greedily packs whole sentences into pieces that each fit the window, falling
back to greedy word packing when a single sentence is itself too long (the
common case for a typed run-on query); no text is dropped. What happens to
the pieces is the **Long-query chunking** setting in the ⚙ dialog:

| Mode | Query vectors | Behaviour |
|---|---|---|
| Truncate | 1 | First 64 tokens only. The old behaviour, kept for comparison. |
| Average | 1 | Every chunk embedded, L2-normalized, averaged. A soft AND — a result has to look somewhat like all of the query. |
| Per chunk *(default)* | N | Each chunk gets its own ranked list; the lists are RRF-fused. A result is rewarded for ranking well against several chunks. |

`AICPreprocess/summary-embed.ipynb` (where `chunk_text` comes from) max-pools
its chunks, and on the corpus side that is right: a summary's chunks are
unrelated topics, so a hit on one is a real hit. On the query side max-pooling
would make added clauses behave like an OR — one strongly-matching clause
could carry a result that ignored the rest — so the per-chunk lists are
**RRF-fused** instead, `1/(RRF_K + rank)` summed over the chunks that
retrieved a row, the same fusion the signals already use across their legs.
That pays a row for placing well against several chunks at once, so the
clauses act as corroborating evidence. Average reaches a similar conjunction
in vector space rather than rank space: blunter (four clauses average into one
point that may sit near none of them), but it keeps a real cosine score. Both
beat Truncate, which just deletes the tail.

Mechanically: `siglip2_query_mat()` returns an `(n_vectors, dim)` matrix —
one row under Truncate/Average, one per chunk under Per chunk — and every
SigLIP2 leg hands it to `common.py::faiss_search_pooled()`, which fuses across
rows. A single-row matrix is passed through to `index.search()` untouched, so
short queries (and image queries, and both single-vector modes) rank
bit-for-bit as they did before any of this existed.

Under Per chunk a leg's `score` is an RRF score (order `1/RRF_K`, so ~0.016
and down) rather than a cosine similarity, and only its order is meaningful.
Nothing downstream is affected — every signal's own RRF fuses on `rank`, not
`score` — but the number on a card is a different scale for a long query than
for a short one. Ties, which are common, break on best cosine and then on
row id, so the ordering is deterministic.

Two more consequences worth knowing:

- The setting is **backend state**, not a browser preference (`GET`/`POST
  /api/settings`) — the splitting and the embedding both happen in the
  backend process, so two tabs on the same port share one value. The dialog
  re-reads it on open rather than trusting its cache. Each profile's process
  has its own, like the profile itself.
- Every leg's TTL cache keys on `common.py::query_hash()`, which folds the
  active strategy in, so a switch shows up on the next search rather than
  being masked by the previous mode's cached ranking.

Only the SigLIP2 legs are affected. The Elasticsearch legs (ASR fuzzy/exact,
caption, OCR, summary fuzzy) always see the raw query string. A long query
still raises a warning banner, kept to one line — the query's token count and
the strategy handling it, e.g. `Query is 128 tokens, over SigLIP2's 64-token
window -- 'chunks_separate'.` Mixed and TRAKE, which can send several texts at
once, list only the ones over the window with their own counts (`#1 (128
tokens)`, `event 2 (77 tokens)`). `siglip2_long_query_note()` builds the
sentence; `siglip2_long_query_tokens()` is the count-only form those two
routes use.

## Signals

| Signal | Legs | Notes |
|---|---|---|
| Keyframe | SigLIP2 only | frame embeddings; CLIP ViT-B/32 + its Multilingual-CLIP query-time text encoder (XLM-RoBERTa-large) were removed entirely -- that text tower alone cost ~4.6GB RAM lazily loaded, dwarfing every other model/index in this system combined |
| ASR | SigLIP2-ASR, Elasticsearch fuzzy, Elasticsearch exact (match_phrase), RRF | transcript segments mapped to nearest keyframe; exact is ASR-only and diacritic-sensitive |
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
