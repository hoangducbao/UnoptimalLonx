# OpticaLynx — C1 Baseline

Minimal end-to-end video moment retrieval baseline (AIC 2026 KIS/VQA path):
index → text query → ranked keyframe results → Streamlit UI. Full spec at
[`docs/c1-baseline-spec.md`](docs/c1-baseline-spec.md). Everything is
embedded/file-based — no server processes.

## Layout

```
pipeline/
  config.py          paths + constants (dataset roots, index paths, backend registry)
  backends.py         registry wiring each backend's paths to its encode_text/load_video
  viclip_encoder.py   ViCLIP-OT text tower (query-time text -> 768-d vector)
  clip_encoder.py      Multilingual-CLIP text tower, paired with CLIP ViT-B/32 image features
  loader.py           per-video frame embeddings + timestamp join, one loader per backend
  detections.py       filtered_detections.csv -> class vocabulary + per-frame labels
  store.py            SQLite metadata (videos, keyframes, classes, frame_classes,
                       keyframe_text FTS5 w/ trigram + diacritic-folded search) —
                       shared across backends
  index_pipeline.py   builds every backend's FAISS indices + the SQLite store
  fusion.py           Reciprocal Rank Fusion (k=60), per-leg weighted — isolated, swappable
  retrieve.py          query pipeline: encode -> search (x2) + FTS5 -> fuse -> enrich;
                       supports restricting every leg to one video
ui/
  app.py              Streamlit UI: backend selector, results grid, click-to-open popup
                       (nearby frames + search-within-this-video)
index/                generated FAISS indices + SQLite DB (git-ignored)
old_version/          the prior CLIP ViT-B/32-only pipeline — reference only, not run
```

## Backends

Two selectable embedding spaces, both indexed over the exact same 873
videos / 177,321 keyframes (verified row-for-row aligned), sharing one
SQLite store and `global_id` scheme — only the FAISS vector files and text
encoder differ per backend (`pipeline/backends.py`):

| Backend | Dim | Frame source | Text encoder |
|---|---|---|---|
| `clip_vitb32` (**default**) | 512 | `AICData/clip-features-32/*.npy` | Multilingual-CLIP |
| `viclip` | 768 | `AICDataExtracted/embeddings/*_viclip768.npy` | ViCLIP-OT (`minhnguyent546/ViCLIP-OT`) |

`clip_vitb32` is the default: measured directly on this dataset, it
discriminates true matches from background noise far more sharply than
`viclip` across the whole corpus (top-1 similarity z-score ≈5.5 vs ≈2.3 for
the same English query, ≈5.4 vs ≈3.5 in Vietnamese), despite `viclip`
producing larger raw similarity numbers. Both stay selectable in the UI
sidebar for comparison.

## Data

- `AICDataExtracted/embeddings/{video_id}_viclip768.npy` + `_filenames.csv` —
  ViCLIP-OT frame embeddings (768-d)
- `AICData/clip-features-32/{video_id}.npy` — CLIP ViT-B/32 frame embeddings
  (512-d, float16 on disk)
- `AICDataExtracted/filtered_detections.csv` — object detections (shared
  across backends; class labels are re-embedded per backend's text tower)
- `AICData/map-keyframes/{video_id}.csv` — reused for per-frame timestamps
  (row-aligned 1:1 with both embedding sources)
- `AICData/keyframes/{video_id}/{filename}` — reused, best-effort, for
  UI thumbnails
- OCR / captions / ASR: not available yet — the corresponding leg (FTS5
  text search) degrades gracefully when empty.

## Query pipeline

Text query → encode with the selected backend's text tower → (1) search
that backend's frame FAISS index, (2) search its object-class FAISS index
and map matches back to frames via the shared `frame_classes` table, (3)
search OCR/caption FTS5 (skipped while empty) → fuse the up-to-three ranked
lists with weighted RRF (`pipeline/fusion.py`) → enrich with
video/timestamp/thumbnail info. Every leg optionally restricts to a single
video (`retrieve.search(..., video_id=...)`), used by the UI's
"search in this video only".

Fusion weights (`config.DEFAULT_LEG_WEIGHTS`, tunable live in the UI
sidebar): the object-class leg is weighted down from 1.0 by default since
its cosine similarities are compressed for compound queries (e.g.
"Motorcycle" 0.63 vs "Candle" 0.61 — barely separated), so a wide top-N
match pool pulls in weakly-related frames if left unweighted.

## UI

`streamlit run ui/app.py` — text query, results grid with per-signal
badges (`frame`/`class`/`text`) showing what contributed to each result's
rank. Click **Open ▸** on any result to open a popup with the surrounding
frames from that video and a "search in this video only" box, which itself
returns results you can click through the same way.

## Usage

```
python pipeline/index_pipeline.py                 # full rebuild, both backends, all videos
python pipeline/index_pipeline.py --limit 5        # quick dev subset
python pipeline/index_pipeline.py --backends viclip  # rebuild just one backend
python pipeline/store.py --stats
python pipeline/retrieve.py "một người đàn ông đang lái xe máy" --backend clip_vitb32
streamlit run ui/app.py
```
