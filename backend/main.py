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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import config
from .es_indexing import ensure_all_fuzzy_indices
from .models import DEVICE, load_siglip2
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

    print("[startup] loading SigLIP2 text/image tower…")
    load_siglip2()

    print("[startup] Keyframe — SigLIP2 frame index")
    kf.build_frame_index(config.FRAME_SIGLIP2_GLOB)
    print("[startup] Keyframe — CLIP frame index")
    kf.build_frame_index(config.FRAME_CLIP_GLOB)

    print("[startup] ASR — SigLIP2 index")
    asr_mod.build_siglip_asr_index()
    print("[startup] Caption — SigLIP2 index")
    cap_mod.build_siglip_caption_index()
    print("[startup] Summary — embeddings + SigLIP2 index")
    sum_mod.build_siglip_summary_index()

    try:
        print("[startup] ASR/Caption/OCR/Summary — Elasticsearch")
        ensure_all_fuzzy_indices()
    except Exception as e:
        print(f"[startup] Elasticsearch warning: {e}")

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search.router)
app.include_router(facets.router)
app.include_router(neighbors.router)
app.include_router(playback.router)
app.include_router(query_image.router)
app.include_router(trake.router)
app.include_router(hierarchy.router)
app.include_router(export.router)

# Media: served directly from the existing AICData* directories, no copying.
config.THUMBNAIL_ROOT.mkdir(parents=True, exist_ok=True)
config.VIDEO_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/media/keyframes", StaticFiles(directory=config.THUMBNAIL_ROOT), name="keyframes")
app.mount("/media/video", StaticFiles(directory=config.VIDEO_DIR), name="video")

# Frontend: static HTML/CSS/JS, served under /app so it doesn't collide
# with /api and /media routes above.
app.mount("/app", NoCacheStaticFiles(directory=config.REPO_ROOT / "frontend", html=True), name="frontend")


@app.get("/")
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/app/")
