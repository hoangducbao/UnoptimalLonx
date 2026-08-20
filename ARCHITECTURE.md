# Architecture & Workflow

## Pipeline

```
raw video corpus (873 videos, ~177k keyframes)
        │
        ▼
external preprocessing (offline, outside this repo)
  ├─ SigLIP2 frame embeddings ─────────► AICDataExtracted/siglib_embed/*.npy
  ├─ CLIP ViT-B/32 frame embeddings ───► AICData/clip-features-32/*.npy
  ├─ ASR transcripts + SigLIP2 embeds ─► AICDataExtracted/transcripts/, asr_embed/
  ├─ frame captions + SigLIP2 embeds ──► AICDataExtracted/captioning/, siglip_caption/
  ├─ OCR text ──────────────────────────► AICDataExtracted/ocr/
  ├─ video summaries ───────────────────► AICDataExtracted/summaries/
  └─ keyframe timestamps + thumbnails ──► AICData/map-keyframes/, keyframes/
        │
        ▼
ui/app.py (single Streamlit process)
  ├─ build FAISS flat-IP indices from the .npy embeddings, cached under index/
  ├─ bulk-index transcript/caption/OCR/summary text into Elasticsearch (fuzzy legs)
  └─ load SigLIP2 + Multilingual-CLIP text towers once, shared across all signals
        │
        ▼
query time: encode query text → search each active leg → per-signal RRF
        → (Mixed mode: weighted RRF across signals) → render_grid()
```

## Signals

| Signal | Legs | Unit |
|---|---|---|
| Keyframe | SigLIP2 + CLIP ViT-B/32, RRF | per frame |
| ASR | SigLIP2-ASR + Elasticsearch fuzzy, RRF | transcript segment → nearest keyframe |
| Caption | SigLIP2-caption + Elasticsearch fuzzy, RRF | per frame |
| OCR | Elasticsearch fuzzy only | per frame |
| Summary | SigLIP2-summary + Elasticsearch fuzzy, RRF | per video |
| Mixed | weighted RRF across Keyframe/ASR/Caption/OCR | per frame |

Every leg normalizes to `{video_id, n, rank, score_label, score_val, text}`
before display, so one `render_grid()` (+ neighbor popup) renders all six
signals.

## Workflow

1. **Offline**: embeddings/text extracted elsewhere, dropped into the
   `AICData*` directory tree (paths hardcoded at the top of `ui/app.py`).
2. **Startup**: `streamlit run ui/app.py` builds/loads every FAISS index
   and ES index eagerly (`st.status` block) before the sidebar renders, and
   loads both text towers once (`st.cache_resource`, process-wide) — this
   must stay a single process to avoid duplicating ~4GB of model weights.
3. **Query**: pick a signal → optional per-leg toggles / weights (Mixed) →
   optional video/collection scope → search → RRF fuse → grid of results,
   each expandable into ±7 neighboring frames.

See `CLAUDE.md` for file-level detail and conventions.
