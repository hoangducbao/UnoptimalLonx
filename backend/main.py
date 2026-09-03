"""
backend/main.py -- FastAPI app entry point. One process serves the JSON
API, the static frontend, and the thumbnail/video media directories --
mirrors ui/app.py's single-process constraint (CLAUDE.md: never duplicate
the ~4GB of loaded model weights across processes), just FastAPI-shaped
instead of Streamlit-shaped.

Run with:
    uvicorn backend.main:app --reload
Then open http://localhost:8000/app/
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from .es_indexing import ensure_all_fuzzy_indices
from .models import (DEVICE, QUERY_CHUNK_STRATEGIES, get_query_chunk_strategy,
                     load_siglip2, set_query_chunk_strategy)
from .routes import export, facets, hierarchy, neighbors, playback, query_image, search, trake
from .search import asr as asr_mod
from .search import caption as cap_mod
from .search import keyframe as kf
from .search import summary as sum_mod


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eager build, once, before the first request is served -- direct
    # replacement for ui/app.py's `with st.status("Loading signals…")`
    # block (ui/app.py:1579-1595).
    config.tune_thread_pools(DEVICE)
    print(f"[startup] device={DEVICE} cpu_budget={config.CPU_BUDGET}")
    # Which embedding profile this process is (backend/config.py). Printed
    # first and loudly: two profiles can run side by side on two ports, and
    # nothing downstream of here says which one you're looking at.
    print(f"[startup] profile={config.EMBED_PROFILE} dim={config.EMBED_DIM} "
          f"model={config.SIGLIP2_MODEL_ID}")
    print(f"[startup] indices={config.INDEX_DIR}")

    print("[startup] loading SigLIP2 text/image tower…")
    load_siglip2()

    # Vector counts per index -- a profile whose upstream embedding job only
    # partly finished still starts and searches fine, just over fewer videos,
    # which is otherwise invisible until rankings look off for no reason.
    print("[startup] Keyframe — SigLIP2 frame index")
    frame_index, frame_lookup = kf.build_frame_index(config.FRAME_SIGLIP2_GLOB)
    print(f"[startup]   {frame_index.ntotal} frames over "
          f"{frame_lookup['video_id'].nunique()} videos")

    print("[startup] ASR — SigLIP2 index")
    asr_index, asr_meta = asr_mod.build_siglip_asr_index()
    print(f"[startup]   {asr_index.ntotal} segments over "
          f"{asr_meta['video_id'].nunique()} videos")
    print("[startup] Caption — SigLIP2 index")
    cap_index, cap_meta = cap_mod.build_siglip_caption_index()
    print(f"[startup]   {cap_index.ntotal} captions over "
          f"{cap_meta['video_id'].nunique()} videos")
    print("[startup] Summary — embeddings + SigLIP2 index")
    sum_index, sum_meta = sum_mod.build_siglip_summary_index()
    print(f"[startup]   {sum_index.ntotal} "
          f"{'chunks' if config.SUMMARY_CHUNKED else 'summaries'} over "
          f"{sum_meta['video_id'].nunique()} videos")

    print("[startup] ASR/Caption/OCR/Summary — Elasticsearch")
    ensure_all_fuzzy_indices()

    print("[startup] all signals ready")
    yield


class NoCacheStaticFiles(StaticFiles):
    """Forces revalidation (not a no-store -- ETag/Last-Modified still let
    a genuinely-unchanged file 304) on every response this mount serves.
    Plain StaticFiles sets no explicit Cache-Control, so a browser's
    default heuristic freshness (RFC 7234 4.2.2, computed from each
    response's own Last-Modified) can keep serving an old JS/CSS file for
    a while after a real edit, with no visible error: the tab just
    silently keeps running stale frontend code against the live
    (already-updated) backend -- this bit us during TRAKE export UI work,
    a tab kept rendering the pre-edit layout with zero indication anything
    was wrong. Used for /app only, not /media -- those files are large and
    genuinely immutable per video_id/frame, unlike frontend source that
    changes underneath an already-open tab during development."""
    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


app = FastAPI(title="Routing101 by MiLF", lifespan=lifespan)

app.include_router(search.router)
app.include_router(facets.router)
app.include_router(neighbors.router)
app.include_router(playback.router)
app.include_router(query_image.router)
app.include_router(trake.router)
app.include_router(hierarchy.router)
app.include_router(export.router)

# Media: served directly from the existing AICData* directories, no copying.
app.mount("/media/keyframes", StaticFiles(directory=config.THUMBNAIL_ROOT), name="keyframes")
app.mount("/media/video", StaticFiles(directory=config.VIDEO_DIR), name="video")

# Frontend: static HTML/CSS/JS, served under /app so it doesn't collide
# with /api and /media routes above.
app.mount("/app", NoCacheStaticFiles(directory=config.REPO_ROOT / "frontend", html=True), name="frontend")


@app.get("/api/profile")
def profile():
    """Which embedding profile this process loaded -- the frontend badges it
    so two tabs on two ports can't be mistaken for each other."""
    return {"profile": config.EMBED_PROFILE, "dim": config.EMBED_DIM,
            "model_id": config.SIGLIP2_MODEL_ID}


class SettingsRequest(BaseModel):
    query_chunk_strategy: str


def _settings_payload():
    return {"query_chunk_strategy": get_query_chunk_strategy(),
            "query_chunk_strategies": list(QUERY_CHUNK_STRATEGIES)}


@app.get("/api/settings")
def get_settings():
    """Backend-side search settings -- currently just how an over-64-token
    query is split for the SigLIP2 embedding legs (backend/models.py). Unlike
    the frontend's own preferences these can't live in localStorage: they
    change what a search returns, and the splitting happens in this process.
    The settings dialog reads this on open so it shows the live value rather
    than whatever the last tab happened to set."""
    return _settings_payload()


@app.post("/api/settings")
def post_settings(body: SettingsRequest):
    try:
        set_query_chunk_strategy(body.query_chunk_strategy)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _settings_payload()


@app.get("/")
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/app/")
