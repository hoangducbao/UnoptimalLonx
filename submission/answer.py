"""submission/answer.py — pluggable Q&A answer generator.

CodaBench QA expects a free-text <Answer> (Vietnamese or English, <=100 chars,
compared by meaning). The retrieval pipeline has no free-text VQA head — that
is a research component to be wired in by the team. This module is the hook.

Configure via SubmissionConfig.answer_mode:
  * "none"     — returns "" (a valid CSV row, just an empty answer). Use for
                 running the end-to-end format pipeline without a VQA model.
  * "caption" / "ocr" — best-effort: pull the matched frame's caption or OCR
                 text out of Elasticsearch and return it as the answer.
  Replace generate_answer() with your VQA/VLM call for a scored run.
"""

from __future__ import annotations

# index-attribute-name -> ES _source field carrying the answer-ish text
_INDEX_MAP = {
    "ocr": ("ES_INDEX_OCR", "text"),
    "caption": ("ES_INDEX_CAPTION", "text"),
}


def _evidence(candidate: dict, mode: str) -> str:
    """Best-effort: the exact caption/OCR text of a matched frame from ES.

    ES doc id convention (see backend/es_indexing.py) is "{video_id}_{frame_id}".
    answer_mode "caption"/"ocr" only light up when ES is reachable.
    """
    try:
        from backend import config as bkc
        from backend.es_client import get_es_client

        index_attr, field = _INDEX_MAP[mode]
        es = get_es_client()
        n = int(candidate["n"])
        resp = es.get(index=getattr(bkc, index_attr), id=f"{candidate['video_id']}_{n}")
        return (resp.get("_source") or {}).get(field, "")
    except Exception:  # noqa: BLE001 — ES down / doc missing -> empty answer
        return ""


def generate_answer(question: str, cfg, candidate: dict | None = None) -> str:
    """Return the <Answer> string for one retrieved candidate frame.

    `candidate` is a {"video_id","n",...} dict from pipeline.mixed_results().
    Replace the body with your model of choice (VQA over the frame thumb, LLM
    over caption/OCR/ASR context, etc.). Keep the return <= cfg.answer_max_len.
    """
    mode = getattr(cfg, "answer_mode", "none")
    if not candidate or mode == "none":
        return ""
    txt = _evidence(candidate, mode)
    if txt:
        return txt[: cfg.answer_max_len]
    return ""