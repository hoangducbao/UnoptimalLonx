# C1 Baseline — Video Moment Retrieval (AIC 2026)

## Goal

Build a minimal end-to-end baseline of the KIS/VQA retrieval path, testable as soon as real data is available. This is a testing scaffold — prioritize simplicity and easy inspection over performance or polish.

Pipeline: index → accept a text query → return ranked keyframe results → show them in a simple UI.

Use current repo's files as references only, when implementing the new just move all old files to "old_version" folder.

---

## Data

Already generated (embeddings, object classes), currently in CSV form in D:\University\Summ26\AICDataExtracted. Logical fields that should exist somewhere across the data (possibly split across multiple files, joined by a frame/video identifier):

- Frame identifier + parent video identifier
- Timestamp within video
- Frame embedding (ViCLIP-OT, 768-d)
- Detected object class labels
- OCR text (not available yet)
- Caption text (not available yet)
- ASR transcript (not available yet)
---

## Storage

No servers — everything embedded/file-based.

| Component | Tech | Purpose |
|---|---|---|
| Vector search | FAISS `IndexFlatIP` — separate indices for frame embeddings and object-class embeddings | Semantic similarity |
| Metadata | SQLite | frame/video/timestamp, join key for everything else |
| Text search | SQLite FTS5, trigram tokenizer, diacritic-folded shadow column | OCR + captions, Vietnamese-aware fuzzy match |

Embeddings must be L2-normalized before insertion. Class labels are embedded via the ViCLIP-OT text tower for cross-language query matching.

Ingestion should be idempotent enough for repeated dev use (wipe-and-rebuild is acceptable) and should tolerate partial data — some fields may be missing until the user fills them in later.

---

## Query pipeline

Given a text query:
1. Encode with ViCLIP-OT, search the frame embedding index
2. Encode with ViCLIP-OT, search the object-class embedding index
3. Search OCR/caption text via FTS5
4. Fuse the (up to three) ranked lists with **Reciprocal Rank Fusion (RRF)**, standard `k=60`, as a placeholder fusion method — isolate this in one clearly swappable function, since the fusion strategy is still under active research and expected to change
5. Return ranked frame results with video/timestamp info

The pipeline must **degrade gracefully** when a signal has no data yet (e.g. class index or FTS5 empty) — skip that leg rather than error, since data will be filled in incrementally.

---

## Interface

Minimal Streamlit UI: text query in, ranked results out. Each result should show enough to identify it (frame/video/timestamp, thumbnail if frame images are available) and ideally which signal(s) contributed to its rank, for debugging fusion behavior. No auth, no persistence beyond the index.

---

## Explicitly out of scope

- Database B, TRAKE, densification, shot-boundary logic
- Any server process (Elasticsearch, Milvus, etc.)

---

## Considering inclusion

- VQA answer generation (LLM-based)
- Relevance feedback / re-ranking UI, auth, deployment

---
