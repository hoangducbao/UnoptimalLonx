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

export function getNeighbors(videoId, centerN, before, after) {
    const qs = new URLSearchParams({ video_id: videoId, center_n: centerN, before, after });
    return jsonFetch(`/api/neighbors?${qs}`);
}

export function getPlayback(videoId, n) {
    const qs = new URLSearchParams({ video_id: videoId, n });
    return jsonFetch(`/api/playback?${qs}`);
}

export async function uploadQueryImage(blob) {
    const form = new FormData();
    form.append("file", blob, "query.png");
    return jsonFetch("/api/query-image", { method: "POST", body: form });
}
