# AICBaseline

The production indexing + retrieval + submission pipeline for the AIC
video-retrieval competition. Reads from `AICData`/`NewData`, builds a
FAISS + SQLite index over CLIP ViT-B/32 features, and answers KIS queries
(VQA/TRAKE planned) against it.

## Layout

```
pipeline/   the actual pipeline, in order (Step 1-4 + query side)
index/      built FAISS index + SQLite DB (generated, not source)
debug/      one-off sanity-check script, not part of the pipeline
```

## `pipeline/`

```
loader.py          Step 1 — load/join one video's per-source files (CLIP .npy, keyframe CSV, media-info, objects)
clean_objects.py    Step 2 — clean raw object-detection JSON into indexable labels
index_pipeline.py   Step 3 — build/extend the FAISS + SQLite index (--incremental default, --rebuild to start over)
store.py            SQLite metadata store, keyed by global_id (== FAISS row id)
text_encoder.py      query string -> 512-d CLIP-space vector (multilingual-CLIP, handles Vietnamese natively)
retrieve.py          query-side KIS search (VQA/TRAKE not yet implemented)
submission.py        writes ranked results to the competition's on-disk submission format
```

Run from anywhere — `--faiss-path`/`--db-path` default to `../index/` relative
to `pipeline/` regardless of your working directory, e.g.:
```
python pipeline/retrieve.py kis "<query>" --faiss-path index/clip_features_flat_ip.index --db-path index/aic_metadata.db
```
(or just omit both flags and let the defaults resolve to `index/`).

## `index/`

`aic_metadata.db` (34MB) + `clip_features_flat_ip.index` (347MB) — the
current, full-dataset build produced by `pipeline/index_pipeline.py`.
Generated, not hand-edited; not required to exist before a fresh
`--rebuild` run.

## `debug/`

`check.py` — a quick standalone sanity check (counts total CLIP vectors
under `AICData/clip-features-32`, independent of the built index). Not
part of the pipeline.
