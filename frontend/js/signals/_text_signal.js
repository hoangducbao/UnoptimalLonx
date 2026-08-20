// frontend/js/signals/_text_signal.js -- ASR/Caption/Summary share one
// shape (SigLIP2 leg + fuzzy leg + RRF, ui/app.py:2000-2050, 2063-2087),
// differing only in labels, the API call, and (Summary) group-by-collection
// instead of group-by-video. This factory avoids repeating that shape
// three times; OCR is the one-leg exception and gets its own file.

import { renderGrid } from "../render.js";
import { currentQuery } from "../query-input.js";

export function makeTextSignalPanel({ prefix, siglipLabel, fuzzyLabel, rrfLabel, searchFn, groupMode }) {
    const controlsHtml = `
    <div class="checkbox-row">
      <input type="checkbox" id="${prefix}-use-siglip" checked>
      <label for="${prefix}-use-siglip" style="margin:0;">${siglipLabel}</label>
    </div>
    <div class="checkbox-row">
      <input type="checkbox" id="${prefix}-use-fuzzy" checked>
      <label for="${prefix}-use-fuzzy" style="margin:0;">${fuzzyLabel}</label>
    </div>
    <div class="checkbox-row">
      <input type="checkbox" id="${prefix}-use-rrf" checked>
      <label for="${prefix}-use-rrf" style="margin:0;">${rrfLabel}</label>
    </div>`;

    function mount(controlsEl) {
        controlsEl.innerHTML = controlsHtml;
    }

    function section(resultsEl, title, leg) {
        const h = document.createElement("h2");
        h.textContent = title;
        resultsEl.append(h);
        const box = document.createElement("div");
        resultsEl.append(box);
        if (!leg) return;
        if (leg.skipped) {
            box.innerHTML = `<div class="status-banner info">${leg.skipped}</div>`;
            return;
        }
        if (leg.warning) {
            const warn = document.createElement("div");
            warn.className = "status-banner warn";
            warn.textContent = leg.warning;
            box.append(warn);
        }
        renderGrid(box, leg.results, groupMode());
    }

    async function run(resultsEl, statusEl) {
        const { query, image_id } = currentQuery();
        if (!query.trim() && !image_id) {
            resultsEl.innerHTML = "";
            statusEl.innerHTML = `<div class="status-banner info">Type a query to search.</div>`;
            return;
        }
        statusEl.innerHTML = "";

        const useSiglip = document.getElementById(`${prefix}-use-siglip`).checked;
        const useFuzzy = document.getElementById(`${prefix}-use-fuzzy`).checked;
        const useRrf = document.getElementById(`${prefix}-use-rrf`).checked;
        if (!useSiglip && !useFuzzy && !useRrf) {
            resultsEl.innerHTML = `<div class="status-banner info">Check at least one search option in the sidebar.</div>`;
            return;
        }

        const topK = parseInt(document.getElementById("top-k").value, 10) || 30;
        const body = {
            query: image_id ? null : query,
            image_id,
            top_k: topK,
            video_filter: document.getElementById("use-video-scope").checked
                ? document.getElementById("video-filter").value : "",
            lot_filter: document.getElementById("use-collection-scope").checked
                ? document.getElementById("lot-filter").value : "",
            legs: { siglip: useSiglip, fuzzy: useFuzzy, rrf: useRrf },
        };

        resultsEl.innerHTML = `<div class="status-banner info">Searching…</div>`;
        let data;
        try {
            data = await searchFn(body);
        } catch (e) {
            resultsEl.innerHTML = `<div class="status-banner error">${e.message}</div>`;
            return;
        }
        resultsEl.innerHTML = "";

        if (useSiglip) section(resultsEl, siglipLabel, data.siglip);
        if (useFuzzy) section(resultsEl, fuzzyLabel, data.fuzzy);
        if (useRrf) section(resultsEl, rrfLabel, data.rrf);
    }

    return { mount, run };
}
