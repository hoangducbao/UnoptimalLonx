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
from fastapi.staticfiles import StaticFiles

from . import config
from .es_indexing import ensure_all_fuzzy_indices
from .models import DEVICE, load_siglip2
from .routes import hierarchy, neighbors, playback, query_image, search, trake
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

    print("[startup] ASR/Caption/OCR/Summary — Elasticsearch")
    ensure_all_fuzzy_indices()

    print("[startup] all signals ready")
    yield


app = FastAPI(title="Routing101 by MiLF", lifespan=lifespan)

app.include_router(search.router)
app.include_router(neighbors.router)
app.include_router(playback.router)
app.include_router(query_image.router)
app.include_router(trake.router)
app.include_router(hierarchy.router)

# Media: served directly from the existing AICData* directories, no copying.
app.mount("/media/keyframes", StaticFiles(directory=config.THUMBNAIL_ROOT), name="keyframes")
app.mount("/media/video", StaticFiles(directory=config.VIDEO_DIR), name="video")

# Frontend: static HTML/CSS/JS, served under /app so it doesn't collide
# with /api and /media routes above.
app.mount("/app", StaticFiles(directory=config.REPO_ROOT / "frontend", html=True), name="frontend")


@app.get("/")
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/app/")
