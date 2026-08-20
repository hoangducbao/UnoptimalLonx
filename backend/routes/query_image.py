"""
backend/routes/query_image.py -- picture-query support. Ported behavior
from ui/app.py's paste-to-image JS trick (ui/app.py:1699-1795): the
frontend posts the pasted/uploaded image bytes here once, gets back a
short-lived `image_id`, then passes that id (instead of `query` text) to
any search endpoint. `resolve_query()` is the shared helper every
search route uses to turn a request body's `query`/`image_id` into the
str-or-PIL.Image object every search_* function already expects
(mirrors ui/app.py's is_image_query/siglip2_query_vec dispatch point).
"""

import io
import uuid

from cachetools import TTLCache
from fastapi import APIRouter, HTTPException, UploadFile
from PIL import Image

router = APIRouter()

# image_id -> PIL.Image, 5-minute TTL -- long enough for a user to paste an
# image and fire a search, short enough not to leak memory over a long
# session (ui/app.py kept the equivalent raw bytes in session_state.query_image_bytes
# for the life of the browser session; a TTL cache is the stateless-backend
# equivalent since there's no per-session server state here).
_IMAGES: TTLCache = TTLCache(maxsize=64, ttl=300)


@router.post("/api/query-image")
async def upload_query_image(file: UploadFile):
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Couldn't decode the pasted image ({e}).")
    image_id = uuid.uuid4().hex
    _IMAGES[image_id] = image
    return {"image_id": image_id}


def resolve_query(query: str | None, image_id: str | None):
    """A picture query takes priority over typed text if both are somehow
    sent (mirrors ui/app.py:1776's `if image_query is not None: query =
    image_query`), matching the frontend's own precedence: a pasted image
    replaces the loaded text query in the UI, not additive."""
    if image_id:
        image = _IMAGES.get(image_id)
        if image is None:
            raise HTTPException(400, "That pasted image has expired -- please paste it again.")
        return image
    if query and query.strip():
        return query
    raise HTTPException(400, "Provide a `query` string or an `image_id`.")
