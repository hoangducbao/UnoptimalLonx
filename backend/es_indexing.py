"""
backend/es_indexing.py -- bulk-index transcript/caption/OCR/summary text
into Elasticsearch. Ported verbatim from ui/app.py's four
ensure_*_fuzzy_index functions (ui/app.py:465-494, 606-630, 691-725,
823-844) -- already idempotent/Streamlit-free logic (checks
es.indices.exists() first, skips the bulk if already indexed). Called once
each from backend/main.py's lifespan, same as the eager `st.status` block
in ui/app.py:1579-1595.
"""

import pandas as pd

from . import config
from .es_client import get_es_client


def ensure_asr_fuzzy_index():
    from elasticsearch import helpers

    es = get_es_client()
    if es.indices.exists(index=config.ES_INDEX_ASR):
        return True
    es.indices.create(index=config.ES_INDEX_ASR, mappings={"properties": {
        "video_id": {"type": "keyword"},
        "segment_id": {"type": "integer"},
        "start_sec": {"type": "float"},
        "text": {"type": "text"},
    }})

    def _docs():
        for csv_path in sorted(config.TRANSCRIPTS_DIR.glob("*.csv")):
            if csv_path.name == "manifest.csv":
                continue
            df = pd.read_csv(csv_path)
            video_id = csv_path.stem
            for _, r in df.iterrows():
                yield {
                    "_index": config.ES_INDEX_ASR,
                    "_id": f"{video_id}_{int(r['segment_id'])}",
                    "_source": {"video_id": video_id, "segment_id": int(r["segment_id"]),
                                 "start_sec": float(r["start_sec"]), "text": r["text"]},
                }

    helpers.bulk(es, _docs(), stats_only=True, raise_on_error=False)
    return True


def ensure_caption_fuzzy_index():
    from elasticsearch import helpers

    es = get_es_client()
    if es.indices.exists(index=config.ES_INDEX_CAPTION):
        return True
    es.indices.create(index=config.ES_INDEX_CAPTION, mappings={"properties": {
        "video_id": {"type": "keyword"},
        "frame_id": {"type": "integer"},
        "text": {"type": "text"},
    }})

    def _docs():
        for csv_path in sorted(config.CAPTIONING_DIR.glob("*.csv")):  # was *_captions.csv before the rename
            if csv_path.name == "manifest.csv":  # run manifest, not a per-video caption file -- different schema entirely
                continue
            df = pd.read_csv(csv_path)
            for _, r in df.iterrows():
                yield {
                    "_index": config.ES_INDEX_CAPTION,
                    "_id": f"{r['video_id']}_{int(r['frame_id'])}",
                    "_source": {"video_id": r["video_id"], "frame_id": int(r["frame_id"]), "text": r["caption_text"]},
                }

    helpers.bulk(es, _docs(), stats_only=True, raise_on_error=False)
    return True


def ensure_ocr_fuzzy_index():
    from elasticsearch import helpers

    es = get_es_client()
    if es.indices.exists(index=config.ES_INDEX_OCR):
        return True
    es.indices.create(index=config.ES_INDEX_OCR, mappings={"properties": {
        "video_id": {"type": "keyword"},
        "frame_id": {"type": "integer"},
        "text": {"type": "text"},
    }})

    def _docs():
        for csv_path in sorted(config.OCR_DIR.glob("*.csv")):
            if csv_path.name.startswith("run_manifest"):
                continue
            video_id = csv_path.stem
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            grouped = df.groupby("frame_id")["text"].apply(
                lambda s: " ".join(str(t) for t in s if pd.notna(t))
            )
            for frame_id, text in grouped.items():
                if not text.strip():
                    continue
                yield {
                    "_index": config.ES_INDEX_OCR,
                    "_id": f"{video_id}_{int(frame_id)}",
                    "_source": {"video_id": video_id, "frame_id": int(frame_id), "text": text},
                }

    helpers.bulk(es, _docs(), stats_only=True, raise_on_error=False)
    return True


def ensure_summary_fuzzy_index():
    from elasticsearch import helpers

    es = get_es_client()
    if es.indices.exists(index=config.ES_INDEX_SUMMARY):
        return True
    es.indices.create(index=config.ES_INDEX_SUMMARY, mappings={"properties": {
        "video_id": {"type": "keyword"},
        "text": {"type": "text"},
    }})

    def _docs():
        for txt_path in sorted(config.SUMMARY_DIR.glob("*.txt")):
            video_id = txt_path.stem
            text = txt_path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            yield {"_index": config.ES_INDEX_SUMMARY, "_id": video_id, "_source": {"video_id": video_id, "text": text}}

    helpers.bulk(es, _docs(), stats_only=True, raise_on_error=False)
    return True


def ensure_all_fuzzy_indices():
    ensure_asr_fuzzy_index()
    ensure_caption_fuzzy_index()
    ensure_ocr_fuzzy_index()
    ensure_summary_fuzzy_index()
