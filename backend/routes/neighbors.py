"""
backend/routes/neighbors.py -- "Show more" nearby-frames popup. Ported
from ui/app.py's show_neighbors dialog (ui/app.py:1334-1360). The dialog's
window (its base size per tile-size setting, plus its expand counters) is
pure frontend state now (plan Decisions section 4), so this endpoint is
stateless: `before`/`after` are the total frames wanted on each side, and
the frontend passes whatever it's currently showing.
"""

from fastapi import APIRouter

from .. import config
from ..common import thumbnail_disk_path, thumbnail_url

router = APIRouter()


@router.get("/api/neighbors")
def get_neighbors(
    video_id: str,
    center_n: int,
    before: int = config.NEIGHBOR_WINDOW,
    after: int = config.NEIGHBOR_WINDOW,
):
    lo = max(0, before)
    hi = max(0, after)
    candidates = [center_n + d for d in range(-lo, hi + 1) if center_n + d >= 1]
    return {
        "video_id": video_id,
        "center_n": center_n,
        "frames": [
            {
                "n": n,
                "thumbnail_url": thumbnail_url(video_id, n),
                "exists": thumbnail_disk_path(video_id, n).exists(),
                "is_center": n == center_n,
            }
            for n in candidates
        ],
    }
