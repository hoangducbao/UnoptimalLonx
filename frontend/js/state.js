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
// Export (AIC submission CSV) -- tracks the current signal's last result
// set, which the standalone Export CSV tab (export-ui.js, opened by
// export-dialog.js) reads as its "similars" preview tier whenever a result
// card's ★ button opens it. Reset on every new search -- a fresh result
// set invalidates the old one.
// ---------------------------------------------------------------------------

export const exportState = {
    candidates: [],  // last search's `results` (flat signals) or `candidates` (TRAKE), as-is
};

export function resetExportCandidates(candidates) {
    exportState.candidates = candidates || [];
}

// Exposed on `window` (not just this module's export binding) so the
// standalone Export CSV tab (frontend/export.html, opened via window.open
// from export-dialog.js) can reach it as `window.opener.__routing101` --
// same-origin windows can read/write each other's plain JS objects
// directly, no postMessage/serialization needed, as long as this tab stays
// open. `exportState` is handed over by reference, so the export tab's
// "Similars" preview reflects this tab's most recent search live, not a
// frozen snapshot from whenever the export tab was opened. `handoffs`
// carries the one-shot trigger object across for each ★ click (see
// export-dialog.js/export-page.js).
window.__routing101 = window.__routing101 || {};
window.__routing101.exportState = exportState;
window.__routing101.handoffs = window.__routing101.handoffs || new Map();

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

// ---------------------------------------------------------------------------
// TRAKE state -- ui/app.py:1810-1812 (trake_context/trake_events/trake_next_id).
// Signal choices offered per event row: every signal except TRAKE itself
// (nested TRAKE makes no sense) and Hierarchy (a grouped/drilled-down
// result set, not the single ranked frame list TRAKE expects per event) --
// ui/app.py:1615.
// ---------------------------------------------------------------------------

// Summary is video-level (one paragraph per video, always resolves to
// frame 1 -- see attach_keyframe_summary in backend/search/summary.py), so
// it's reserved for the context row's whole-video boost query and left out
// of the per-event signal choices below (an ordered event pinned to frame
// 1 mostly just breaks TRAKE's strict-order check anyway).
export const TRAKE_EVENT_SIGNALS = ["Keyframe", "ASR", "Caption", "OCR", "Mixed"];

export const trakeState = {
    // Context's signal is fixed to "Summary" -- no dropdown, see trake.js's
    // context row -- not one of the user-facing TRAKE_EVENT_SIGNALS above.
    context: { text: "", signal: "Summary" },
    events: [{ id: 0, text: "", signal: "Keyframe" }],
    nextId: 1,
};

// ---------------------------------------------------------------------------
// Mixed search state -- many independent sub-queries, each with its own
// text + single signal (no nested "Mixed", no image sub-queries -- unlike
// TRAKE's events, these aren't ordered and aren't required to all match).
// Combined via weighted RRF, weight (0-3) chosen per sub-query below.
// ---------------------------------------------------------------------------

// Summary excluded here too, same reasoning as TRAKE_EVENT_SIGNALS above.
export const MIXED_QUERY_SIGNALS = ["Keyframe", "ASR", "Caption", "OCR"];

export const mixedQueryState = {
    queries: [{ id: 0, text: "", signal: "Keyframe", weight: 1 }],
    nextId: 1,
};

// Hierarchy: per-video Top-G override, mirrors ui/app.py's
// session_state.hier_extra_g (ui/app.py:2136-2137) -- the "Expand" button
// bumps just that one video's effective G by +10, independent of every
// other group and of the sidebar's Top-G control.
export const hierExtraG = new Map(); // video_id -> extra G (multiples of 10)

export function getNeighborExtra(videoId, centerN) {
    const key = `${videoId}_${centerN}`;
    if (!state.neighborExtra.has(key)) {
        state.neighborExtra.set(key, { before: 0, after: 0 });
    }
    return state.neighborExtra.get(key);
}

function scopeToggleActive(scope) {
    return document.querySelector(`#scope-segmented button[data-scope="${scope}"]`).classList.contains("active");
}

// Reads the sidebar's video/collection scope controls into a request-body
// fragment -- shared by every signal's search-body builder so the "excl"
// toggle (drop the collection range instead of restricting to it) only has
// to be wired up here, not independently in six signal files.
// "excl" works whether or not "coll" is also on: either one alone is
// enough to send the collection range, and "excl" decides which way it's
// applied -- so a bare "excl" (no "coll") still drops that range instead
// of being a silent no-op.
export function scopeFilters() {
    const collActive = scopeToggleActive("collection");
    const exclActive = scopeToggleActive("exclude");
    return {
        video_filter: scopeToggleActive("video")
            ? document.getElementById("video-filter").value : "",
        lot_filter: (collActive || exclActive)
            ? document.getElementById("lot-filter").value : "",
        exclude_lot: exclActive,
        od_filter: document.getElementById("od-filter").value,
        facet_field: document.getElementById("facet-field").value,
        facet_value: document.getElementById("facet-field").value
            ? document.getElementById("facet-value").value : "",
    };
}

// Best-effort OS clipboard write, alongside the scope-box fill below --
// wrapped since Clipboard.writeText() can reject (no secure context, no
// document focus, permission denied) and that must never break the
// scope-fill behavior it rides along with.
function copyTextToClipboard(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).catch(() => {});
    }
}

// Mirrors ui/app.py's copy_to_scope/copy_collection_only (ui/app.py:233-242):
// fills the scope boxes from one frame's video_id, but -- same as the
// original -- does NOT auto-check "Use video"/"Use collection" for you.
export function copyToScope(videoId) {
    document.getElementById("video-filter").value = videoId;
    const m = /^L(\d+)/i.exec(videoId);
    document.getElementById("lot-filter").value = m ? `L${m[1]}` : videoId;
    copyTextToClipboard(videoId);
}

export function copyCollectionOnly(videoId) {
    const m = /^L(\d+)/i.exec(videoId);
    document.getElementById("lot-filter").value = m ? `L${m[1]}` : videoId;
    copyTextToClipboard(videoId);
}
