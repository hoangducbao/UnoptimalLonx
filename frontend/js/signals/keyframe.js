// frontend/js/signals/keyframe.js -- Keyframe signal panel: SigLIP2-only
// search + render. CLIP ViT-B/32 (and its Multilingual-CLIP query-time
// text encoder) was removed entirely, so this signal has no legs to
// choose and no RRF to fuse -- a single flat result grid, same shape as
// OCR/Caption's own single-leg sections.

import { searchKeyframe } from "../api.js";
import { renderGrid } from "../render.js";
import { currentQuery } from "../query-input.js";
import { resetExportCandidates, scopeFilters } from "../state.js";
import { settings } from "../settings.js";

export function mount(controlsEl) {
    controlsEl.innerHTML = "";
}

function groupMode() {
    return settings.groupByVideo ? "video" : null;
}

export async function run(resultsEl, statusEl) {
    const { query, image_id } = currentQuery();
    if (!query.trim() && !image_id) {
        resultsEl.innerHTML = "";
        statusEl.innerHTML = `<div class="status-banner info">Type a query to search.</div>`;
        return;
    }
    statusEl.innerHTML = "";

    const topK = parseInt(document.getElementById("top-k").value, 10) || 100;
    const body = {
        query: image_id ? null : query,
        image_id,
        top_k: topK,
        ...scopeFilters(),
    };

    resultsEl.innerHTML = `<div class="status-banner info">Searching…</div>`;
    let data;
    try {
        data = await searchKeyframe(body);
    } catch (e) {
        resultsEl.innerHTML = `<div class="status-banner error">${e.message}</div>`;
        return;
    }
    resultsEl.innerHTML = "";
    resetExportCandidates(data.results);

    const box = document.createElement("div");
    resultsEl.append(box);
    // renderGrid() does container.innerHTML = "" first -- render before
    // appending the warning, or it gets wiped out with everything else.
    renderGrid(box, data.results, groupMode());
    if (data.warning) {
        const warn = document.createElement("div");
        warn.className = "status-banner warn";
        warn.textContent = data.warning;
        box.prepend(warn);
    }
}
