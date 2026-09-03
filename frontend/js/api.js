// frontend/js/api.js -- thin fetch() wrappers, one per backend endpoint.

async function jsonFetch(url, options) {
    const res = await fetch(url, options);
    if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `${res.status} ${res.statusText}`);
    }
    return res.json();
}

function postJson(url, body) {
    return jsonFetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
}

export const searchKeyframe = (body) => postJson("/api/search/keyframe", body);
export const searchAsr = (body) => postJson("/api/search/asr", body);
export const searchCaption = (body) => postJson("/api/search/caption", body);
export const searchOcr = (body) => postJson("/api/search/ocr", body);
export const searchSummary = (body) => postJson("/api/search/summary", body);
export const searchMixed = (body) => postJson("/api/search/mixed", body);
export const searchTrake = (body) => postJson("/api/search/trake", body);
export const searchHierarchy = (body) => postJson("/api/search/hierarchy", body);
export const expandHierarchy = (body) => postJson("/api/hierarchy/expand", body);
export const getFacets = () => jsonFetch("/api/facets");
export const getProfile = () => jsonFetch("/api/profile");

// Backend-side search settings (query chunking). Separate from settings.js's
// localStorage preferences on purpose: this one changes what a search
// returns and is applied inside the backend process, so it can't be a
// per-browser value -- see backend/main.py's /api/settings.
export const getSearchSettings = () => jsonFetch("/api/settings");
export const setSearchSettings = (body) => postJson("/api/settings", body);

// before/after are the totals wanted on each side of centerN (base window
// per tile size + however far the popup has been expanded), not deltas.
export function getNeighbors(videoId, centerN, before, after) {
    const qs = new URLSearchParams({ video_id: videoId, center_n: centerN, before, after });
    return jsonFetch(`/api/neighbors?${qs}`);
}

// n omitted (undefined/null) starts playback at 0:00 with no keyframe
// lookup -- used by the TRAKE Export tab's curation panel to play a bare
// video_id before any event exists yet (see backend/routes/playback.py).
export function getPlayback(videoId, n) {
    const params = { video_id: videoId };
    if (n !== undefined && n !== null) params.n = n;
    const qs = new URLSearchParams(params);
    return jsonFetch(`/api/playback?${qs}`);
}

export function getExportFrame(videoId, n) {
    const qs = new URLSearchParams({ video_id: videoId, n });
    return jsonFetch(`/api/export/frame?${qs}`);
}

export function getExportNeighbors(videoId, n, count) {
    const qs = new URLSearchParams({ video_id: videoId, n, count });
    return jsonFetch(`/api/export/neighbors?${qs}`);
}

// Confirmed mode's "Similars" preview -- a fresh visual search seeded by
// the confirmed frame itself (see backend/export.py's
// similar_candidates_for_frame), not the opener tab's original query
// results. Same {video_id, n, results} shape as getExportNeighbors above.
export function getExportSimilar(videoId, n, count) {
    const qs = new URLSearchParams({ video_id: videoId, n, count });
    return jsonFetch(`/api/export/similar?${qs}`);
}

// Export tab's "Keyframes" checkbox, re-checked while a raw native frame
// (Keyframes unchecked) is curated -- snaps that frame_idx to its nearest
// indexed keyframe so keyframe-mode UI has an n again (see
// backend/export.py::similar_candidates_for_native_frame's docstring for
// the same snap-to-nearest idea, used server-side for the CSV itself).
export function getExportNearestKeyframe(videoId, frameIdx) {
    const qs = new URLSearchParams({ video_id: videoId, frame_idx: frameIdx });
    return jsonFetch(`/api/export/nearest-keyframe?${qs}`);
}

export async function uploadQueryImage(blob) {
    const form = new FormData();
    form.append("file", blob, "query.png");
    return jsonFetch("/api/query-image", { method: "POST", body: form });
}

// Export returns CSV text, not JSON, so it can't go through jsonFetch --
// fetch it as a blob and trigger a browser download via a throwaway <a>.
// Shared by exportCsv (KIS/VQA) and writeTrakeCsv (TRAKE) below, both of
// which hit a PlainTextResponse+Content-Disposition endpoint.
async function postForCsvDownload(url, body) {
    const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.detail || `${res.status} ${res.statusText}`);
    }
    const blob = await res.blob();
    const match = /filename="?([^"]+)"?/.exec(res.headers.get("Content-Disposition") || "");
    const filename = match ? match[1] : "export.csv";

    const objUrl = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = objUrl;
    a.download = filename;
    document.body.append(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(objUrl);
}

export const exportCsv = (body) => postForCsvDownload("/api/export", body);

// TRAKE row generation: one video's curated ordered frame_idx list ->
// <=max_rows candidate sequences, cached client-side (export-ui.js) keyed
// by video_id. Pure computation, no download -- see backend/export.py's
// generate_trake_rows().
export function getTrakeRows(videoId, frameIdxs, maxRows) {
    return postJson("/api/export/trake-rows", { video_id: videoId, frame_idxs: frameIdxs, max_rows: maxRows });
}

// TRAKE final export: a human-merged row set, already interleaved
// client-side from the per-video cache -- no ranking/dedup left to do,
// just CSV formatting + download.
export const writeTrakeCsv = (rows, filename) => postForCsvDownload("/api/export/trake-write", { rows, filename });
