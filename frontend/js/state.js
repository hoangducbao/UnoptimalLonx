// frontend/js/state.js -- client-side state, the direct equivalent of
// Streamlit's per-session `session_state` (see the rewrite plan's
// Decisions section 4) -- plain in-memory JS, scoped to the page session,
// reset on reload. Grows one field per phase as each signal is wired up.

export const state = {
    signal: "Keyframe",
    imageQueryId: null,       // set by query-input.js's paste handler
    // Per-{video_id}_{center_n} neighbor-popup expand counters, mirrors
    // ui/app.py's session_state[f"nbr_extra_{video_id}_{center_n}"].
    neighborExtra: new Map(), // key `${videoId}_${centerN}` -> {before, after}
};

export function getNeighborExtra(videoId, centerN) {
    const key = `${videoId}_${centerN}`;
    if (!state.neighborExtra.has(key)) {
        state.neighborExtra.set(key, { before: 0, after: 0 });
    }
    return state.neighborExtra.get(key);
}

// Mirrors ui/app.py's copy_to_scope/copy_collection_only (ui/app.py:233-242):
// fills the scope boxes from one frame's video_id, but -- same as the
// original -- does NOT auto-check "Use video"/"Use collection" for you.
export function copyToScope(videoId) {
    document.getElementById("video-filter").value = videoId;
    const m = /^L(\d+)/i.exec(videoId);
    document.getElementById("lot-filter").value = m ? `L${m[1]}` : videoId;
}

export function copyCollectionOnly(videoId) {
    const m = /^L(\d+)/i.exec(videoId);
    document.getElementById("lot-filter").value = m ? `L${m[1]}` : videoId;
}
