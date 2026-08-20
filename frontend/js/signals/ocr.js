// frontend/js/signals/ocr.js -- OCR signal panel: single leg by design, no
// embedding leg, no RRF, no leg checkboxes (ui/app.py:1962-1963, 2052-2061).

import { searchOcr } from "../api.js";
import { renderGrid } from "../render.js";
import { currentQuery } from "../query-input.js";

export function mount(controlsEl) {
    controlsEl.innerHTML = `<div class="thumb-caption muted">Single leg: fuzzy text search only, no embedding leg.</div>`;
}

function groupMode() {
    return document.getElementById("group-by-video").checked ? "video" : null;
}

export async function run(resultsEl, statusEl) {
    const { query, image_id } = currentQuery();
    if (!query.trim() && !image_id) {
        resultsEl.innerHTML = "";
        statusEl.innerHTML = `<div class="status-banner info">Type a query to search.</div>`;
        return;
    }
    statusEl.innerHTML = "";

    const topK = parseInt(document.getElementById("top-k").value, 10) || 30;
    const body = {
        query: image_id ? null : query,
        image_id,
        top_k: topK,
        video_filter: document.getElementById("use-video-scope").checked
            ? document.getElementById("video-filter").value : "",
        lot_filter: document.getElementById("use-collection-scope").checked
            ? document.getElementById("lot-filter").value : "",
    };

    resultsEl.innerHTML = `<div class="status-banner info">Searching…</div>`;
    let data;
    try {
        data = await searchOcr(body);
    } catch (e) {
        resultsEl.innerHTML = `<div class="status-banner error">${e.message}</div>`;
        return;
    }
    resultsEl.innerHTML = "";

    if (data.image_query_unavailable) {
        resultsEl.innerHTML = `<div class="status-banner info">OCR is fuzzy text search only — not available for picture queries.</div>`;
        return;
    }

    const h = document.createElement("h2");
    h.textContent = "Fuzzy OCR";
    resultsEl.append(h);
    const box = document.createElement("div");
    resultsEl.append(box);
    if (data.fuzzy.warning) {
        const warn = document.createElement("div");
        warn.className = "status-banner warn";
        warn.textContent = data.fuzzy.warning;
        box.append(warn);
    }
    renderGrid(box, data.fuzzy.results, groupMode());
}
