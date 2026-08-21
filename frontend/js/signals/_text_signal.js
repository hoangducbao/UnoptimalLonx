// frontend/js/signals/_text_signal.js -- ASR/Caption/Summary share one
// shape (SigLIP2 leg + fuzzy leg + RRF, ui/app.py:2000-2050, 2063-2087),
// differing only in labels, the API call, (Summary) group-by-collection
// instead of group-by-video, and (per-instance) checkbox order/defaults.
// This factory avoids repeating that shape three times; OCR is the
// one-leg exception and gets its own file.

import { renderGrid } from "../render.js";
import { currentQuery } from "../query-input.js";

const LEG_KEYS = ["siglip", "fuzzy", "rrf"];

export function makeTextSignalPanel({
    prefix, siglipLabel, fuzzyLabel, rrfLabel, searchFn, groupMode,
    order = LEG_KEYS, defaults = { siglip: true, fuzzy: true, rrf: true },
}) {
    const labels = { siglip: siglipLabel, fuzzy: fuzzyLabel, rrf: rrfLabel };
    const controlsHtml = order.map((key) => `
    <div class="checkbox-row">
      <input type="checkbox" id="${prefix}-use-${key}"${defaults[key] ? " checked" : ""}>
      <label for="${prefix}-use-${key}" style="margin:0;">${labels[key]}</label>
    </div>`).join("");

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

        const use = Object.fromEntries(LEG_KEYS.map((key) => [key, document.getElementById(`${prefix}-use-${key}`).checked]));
        if (!use.siglip && !use.fuzzy && !use.rrf) {
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
            legs: { siglip: use.siglip, fuzzy: use.fuzzy, rrf: use.rrf },
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

        const dataKey = { siglip: "siglip", fuzzy: "fuzzy", rrf: "rrf" };
        for (const key of order) {
            if (use[key]) section(resultsEl, labels[key], data[dataKey[key]]);
        }
    }

    return { mount, run };
}
