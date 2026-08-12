"""
fusion.py — combine up to three per-signal ranked lists into one ranked
result list.

This is a placeholder fusion method (standard Reciprocal Rank Fusion,
k=60) per docs/c1-baseline-spec.md -- fusion strategy is still under active
research and expected to change, so it's isolated in this one function and
nothing else in the pipeline should know how fusion works internally.
Graceful degradation when a leg has no data resolves here: legs simply
absent from `ranked_lists` (or with an empty list) contribute nothing.
"""

from collections import defaultdict

import config


def reciprocal_rank_fusion(ranked_lists: dict, k: int = config.RRF_K, weights: dict | None = None) -> list:
    """ranked_lists: {leg_name: [global_id, ...]} best-first, for whichever
    legs actually produced results.

    weights: optional {leg_name: float} multiplier on each leg's contribution
    (default 1.0 for any leg not listed) -- lets a noisier leg be counted for
    less without excluding it outright. Defaults to config.DEFAULT_LEG_WEIGHTS.

    Returns a list of (global_id, rrf_score, breakdown) sorted by rrf_score
    descending, where breakdown = {leg_name: {"rank": int, "contribution": float}}
    for every leg that included that global_id -- used by the UI to show
    which signal(s) contributed to a result's rank."""
    weights = config.DEFAULT_LEG_WEIGHTS if weights is None else weights
    scores: dict = defaultdict(float)
    breakdown: dict = defaultdict(dict)

    for leg_name, ids in ranked_lists.items():
        weight = weights.get(leg_name, 1.0)
        for rank, global_id in enumerate(ids or []):
            contribution = weight / (k + rank + 1)
            scores[global_id] += contribution
            breakdown[global_id][leg_name] = {"rank": rank, "contribution": contribution}

    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(global_id, score, breakdown[global_id]) for global_id, score in fused]
