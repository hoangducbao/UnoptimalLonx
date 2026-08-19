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
  config.py          paths + constants shared by routing101.py and the text encoders
  clip_encoder.py     Multilingual-CLIP text tower, paired with CLIP ViT-B/32 image features
  viclip_encoder.py   ViCLIP-OT text tower (routing101.py --backend viclip)
  loader.py           per-video frame embeddings + timestamp join (routing101.py)
ui/
  app.py              combined Streamlit UI — all three layers below, one process
index/                generated FAISS indices (git-ignored), reused across app runs
routing101.py          CLI: single-embedding-space text -> frame search (see its docstring)
routing101.ipynb        annotated walkthrough: keyframe embeddings (SigLIP2 / CLIP ViT-B/32 / RRF)
routing101_asr.ipynb     annotated walkthrough: ASR-segment search + RRF, mapped to keyframes
routing101_caption.ipynb annotated walkthrough: frame-caption search + RRF, mapped to keyframes
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
docker run -d --name es -p 9200:9200 -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" docker.elastic.co/elasticsearch/elasticsearch:8.15.0
```

If ES isn't reachable, the fuzzy leg degrades to an empty result (with an
on-page warning) rather than breaking the other legs.

## CLI

```
python routing101.py "a man riding a motorbike" --backend siglip2 --k 100
python routing101.py --eval queries.csv --backend clip_vitb32
```

See `routing101.py`'s module docstring for the three selectable backends
(`siglip2`, `clip_vitb32`, `viclip`) and their on-disk index caching.
