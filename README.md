# OpticaLynx — C1 Baseline

Minimal end-to-end video moment retrieval baseline (AIC 2026 KIS/VQA path):
index → text query → ranked keyframe results → Streamlit UI. Full spec at
[`docs/c1-baseline-spec.md`](docs/c1-baseline-spec.md). Everything is
embedded/file-based — no server processes.

## Layout

```
pipeline/
  config.py          paths + constants (dataset roots, index paths, model id)
  viclip_encoder.py  ViCLIP-OT text tower (query-time text -> 768-d vector)
  loader.py          per-video frame embeddings + timestamp join
  detections.py      filtered_detections.csv -> class vocabulary + per-frame labels
  store.py           SQLite metadata (videos, keyframes, classes, frame_classes,
                      keyframe_text FTS5 w/ trigram + diacritic-folded search)
  index_pipeline.py  builds both FAISS indices + the SQLite store
  fusion.py          Reciprocal Rank Fusion (k=60) — isolated, swappable
  retrieve.py         query pipeline: encode -> search (x2) + FTS5 -> fuse -> enrich
ui/
  app.py             Streamlit UI
index/               generated FAISS indices + SQLite DB (git-ignored)
old_version/         the prior CLIP ViT-B/32 pipeline — reference only, not run
```

## Data

- `AICDataExtracted/embeddings/{video_id}_viclip768.npy` + `_filenames.csv` —
  ViCLIP-OT frame embeddings (768-d)
- `AICDataExtracted/filtered_detections.csv` — object detections
- `AICData/map-keyframes/{video_id}.csv` — reused for per-frame timestamps
  (row-aligned 1:1 with the embeddings)
- `AICData/keyframes/{video_id}/{filename}` — reused, best-effort, for
  UI thumbnails
- OCR / captions / ASR: not available yet — the corresponding legs
  (object-class index, FTS5 text search) degrade gracefully when empty.

## Usage

```
python pipeline/index_pipeline.py                 # full rebuild, all videos
python pipeline/index_pipeline.py --limit 5        # quick dev subset
python pipeline/store.py --stats
python pipeline/retrieve.py "một người đàn ông đang lái xe máy"
streamlit run ui/app.py
```
