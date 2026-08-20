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

// ---------------------------------------------------------------------------
// Mixed mode config -- ui/app.py:1252-1267. ONE shared config, read/written
// from standalone Mixed mode AND every TRAKE row set to "Mixed" (a later
// phase) -- same single-global-dict coupling as the original, see the
// rewrite plan's Decisions section 2. Persisted to localStorage: a small
// superset of ui/app.py's per-session behavior (survives reloads too),
// not a limitation.
// ---------------------------------------------------------------------------

export const MIXED_SIGNAL_NAMES = ["Keyframe", "ASR", "Caption", "OCR"];
export const MIXED_LEG_DEFS = {
    Keyframe: [["kf_siglip2", "SigLIP2"], ["kf_clip", "CLIP"]],
    ASR: [["asr_siglip", "SigLIP2 ASR"], ["asr_fuzzy", "Fuzzy ASR"]],
    Caption: [["cap_siglip", "SigLIP2 Caption"], ["cap_fuzzy", "Fuzzy Caption"]],
    // OCR intentionally omitted -- single fuzzy-only leg, nothing to choose.
};
export const MIXED_DEFAULT_WEIGHTS = Object.fromEntries(MIXED_SIGNAL_NAMES.map((n) => [n, 1]));
export const MIXED_DEFAULT_LEGS = {
    kf_siglip2: true, kf_clip: true,
    asr_siglip: false, asr_fuzzy: true,
    cap_siglip: false, cap_fuzzy: true,
};

const MIXED_STORAGE_KEY = "routing101_mixed_config";

function loadMixedConfig() {
    try {
        const raw = localStorage.getItem(MIXED_STORAGE_KEY);
        if (raw) {
            const parsed = JSON.parse(raw);
            return {
                weights: { ...MIXED_DEFAULT_WEIGHTS, ...parsed.weights },
                legs: { ...MIXED_DEFAULT_LEGS, ...parsed.legs },
            };
        }
    } catch (e) { /* corrupt/old value -- fall through to defaults */ }
    return { weights: { ...MIXED_DEFAULT_WEIGHTS }, legs: { ...MIXED_DEFAULT_LEGS } };
}

export const mixedConfig = loadMixedConfig();

export function saveMixedConfig() {
    localStorage.setItem(MIXED_STORAGE_KEY, JSON.stringify(mixedConfig));
}

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
