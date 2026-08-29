// frontend/js/signals/keyframe.js -- Keyframe signal panel: leg checkboxes
// + search + render. Direct port of ui/app.py:1950-1953 (sidebar controls)
// and ui/app.py:1972-1998 (render block).

import { searchKeyframe } from "../api.js";
import { renderGrid } from "../render.js";
import { currentQuery } from "../query-input.js";
import { resetExportCandidates, scopeFilters } from "../state.js";

const controlsHtml = `
<div class="checkbox-row">
  <input type="checkbox" id="kf-use-siglip2" checked>
  <label for="kf-use-siglip2" style="margin:0;">SigLIP2 Frame</label>
</div>
<div class="checkbox-row">
  <input type="checkbox" id="kf-use-clip" checked>
  <label for="kf-use-clip" style="margin:0;">CLIP Frame</label>
</div>
<div class="checkbox-row">
  <input type="checkbox" id="kf-use-rrf" checked>
  <label for="kf-use-rrf" style="margin:0;">RRF Frame</label>
</div>`;

export function mount(controlsEl) {
    controlsEl.innerHTML = controlsHtml;
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

    const useSiglip2 = document.getElementById("kf-use-siglip2").checked;
    const useClip = document.getElementById("kf-use-clip").checked;
    const useRrf = document.getElementById("kf-use-rrf").checked;
    if (!useSiglip2 && !useClip && !useRrf) {
        resultsEl.innerHTML = `<div class="status-banner info">Check at least one search option in the sidebar.</div>`;
        return;
    }

    const topK = parseInt(document.getElementById("top-k").value, 10) || 100;
    const body = {
        query: image_id ? null : query,
        image_id,
        top_k: topK,
        ...scopeFilters(),
        legs: { siglip2: useSiglip2, clip: useClip, rrf: useRrf },
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
    const gm = groupMode();

    // Export candidates come from the "best" enabled leg: RRF (fused) if
    // on, else whichever plain leg is -- same priority as _text_signal.js.
    const exportLeg = (data.rrf && !data.rrf.skipped && useRrf) ? data.rrf
        : (data.siglip2 && useSiglip2) ? data.siglip2
        : (data.clip && !data.clip.skipped && useClip) ? data.clip
        : null;
    resetExportCandidates(exportLeg ? exportLeg.results : []);

    section(resultsEl, "SigLIP2", data.siglip2, gm);
    section(resultsEl, "CLIP", data.clip, gm);
    section(resultsEl, "RRF", data.rrf, gm);
}

function section(resultsEl, title, leg, gm) {
    if (!leg) return;
    const h = document.createElement("h2");
    h.textContent = title;
    resultsEl.append(h);
    const box = document.createElement("div");
    resultsEl.append(box);
    if (leg.skipped) {
        box.innerHTML = `<div class="status-banner info">${leg.skipped}</div>`;
        return;
    }
    // renderGrid() does container.innerHTML = "" first -- render before
    // appending the warning, or it gets wiped out with everything else.
    renderGrid(box, leg.results, gm);
    if (leg.warning) {
        const warn = document.createElement("div");
        warn.className = "status-banner warn";
        warn.textContent = leg.warning;
        box.prepend(warn);
    }
}
