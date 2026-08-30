"""
backend/routes/playback.py -- video streaming and playback metadata endpoint.
Supports:
- HTTP 206 Partial Content (Byte Range Requests) for HTML5 <video> scrubbing.
- Smart auto-discovery of video files across lot folders and Kaggle dataset inputs (e.g. degarr).
- Rich keyframe metadata and fallback descriptors when raw video is missing.
"""

import os
from typing import Optional
from pathlib import Path
import mimetypes

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse, Response

from .. import config
from ..common import (
    find_video_path,
    get_video_keyframes_meta,
    keyframe_timestamp,
    video_fps_for_video,
    video_url,
)

router = APIRouter()


@router.get("/api/playback")
def get_playback(video_id: str, n: Optional[int] = None):
    vpath = find_video_path(video_id)
    has_video = vpath is not None and vpath.exists()
    
    if n is None:
        ts, fps = 0, video_fps_for_video(video_id)
    else:
        ts, fps = keyframe_timestamp(video_id, n)

    keyframes = get_video_keyframes_meta(video_id)
    
    return {
        "video_id": video_id,
        "has_video": has_video,
        "video_url": video_url(video_id),
        "start_time": ts if ts is not None else 0,
        "fps": fps if fps is not None else 25.0,
        "total_keyframes": len(keyframes),
        "keyframes": keyframes,
    }


def _file_chunk_generator(file_path: Path, start: int, chunk_size: int, block_size: int = 1024 * 1024):
    with open(file_path, "rb") as f:
        f.seek(start)
        remaining = chunk_size
        while remaining > 0:
            bytes_to_read = min(remaining, block_size)
            data = f.read(bytes_to_read)
            if not data:
                break
            remaining -= len(data)
            yield data


@router.get("/api/playback/stream/{video_id}")
async def stream_video(video_id: str, request: Request, range: Optional[str] = Header(None)):
    vpath = find_video_path(video_id)
    if not vpath or not vpath.exists():
        raise HTTPException(404, f"Video stream not found for {video_id}.")

    file_size = vpath.stat().st_size
    mime_type, _ = mimetypes.guess_type(str(vpath))
    content_type = mime_type or "video/mp4"

    # Handle Byte-Range Request (HTTP 206)
    if range:
        try:
            # Parse 'bytes=start-end'
            range_str = range.strip()
            if range_str.startswith("bytes="):
                range_str = range_str[6:]
            parts = range_str.split("-")
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
            start = max(0, min(start, file_size - 1))
            end = max(start, min(end, file_size - 1))
            chunk_size = (end - start) + 1

            headers = {
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(chunk_size),
                "Content-Type": content_type,
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Range, Accept-Ranges, Content-Range",
            }
            return StreamingResponse(
                _file_chunk_generator(vpath, start, chunk_size),
                status_code=206,
                headers=headers,
            )
        except Exception:
            pass

    # Full content response with Accept-Ranges support
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(file_size),
        "Content-Type": content_type,
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Range, Accept-Ranges, Content-Range",
    }
    return StreamingResponse(
        _file_chunk_generator(vpath, 0, file_size),
        status_code=200,
        headers=headers,
    )
