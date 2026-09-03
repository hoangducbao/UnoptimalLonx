// frontend/js/signals/_text_signal.js -- ASR/Caption/Summary share one
// shape (SigLIP2 leg + fuzzy leg + RRF, ui/app.py:2000-2050, 2063-2087),
// differing only in labels, the API call, (Summary) group-by-collection
// instead of group-by-video, (per-instance) checkbox order/defaults, and
// which legs exist at all -- ASR adds a fourth, Exact.
// This factory avoids repeating that shape three times; OCR is the
// one-leg exception and gets its own file.

import { renderGrid } from "../render.js";
import { currentQuery } from "../query-input.js";
import { resetExportCandidates, scopeFilters } from "../state.js";

// A panel's `order` is the authoritative list of the legs it actually has --
// it drives the checkboxes, the request, and the render, so a panel that
// omits a leg never looks for its checkbox. ASR is the only one that opts
// into `exact`; Caption/Summary keep the original three.
const DEFAULT_ORDER = ["siglip", "fuzzy", "rrf"];
// Export candidates come from the single "best" enabled leg, not every leg
// rendered on screen -- RRF (fused) is preferred when enabled, since it's
// the strongest ranking available, falling back to whichever plain leg is
// on when RRF isn't. Filtered against `order` per panel below.
const EXPORT_LEG_PRIORITY = ["rrf", "siglip", "exact", "fuzzy"];

export function makeTextSignalPanel({
    prefix, siglipLabel, fuzzyLabel, exactLabel, rrfLabel, searchFn, groupMode,
    order = DEFAULT_ORDER, defaults = { siglip: true, fuzzy: true, rrf: true },
}) {
    const labels = { siglip: siglipLabel, fuzzy: fuzzyLabel, exact: exactLabel, rrf: rrfLabel };
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
        // renderGrid() does container.innerHTML = "" first -- render the
        // grid before appending the warning, not after, or the warning gets
        // wiped out along with everything else already in `box`.
        renderGrid(box, leg.results, groupMode());
        if (leg.warning) {
            const warn = document.createElement("div");
            warn.className = "status-banner warn";
            warn.textContent = leg.warning;
            box.prepend(warn);
        }
    }

    async function run(resultsEl, statusEl) {
        const { query, image_id } = currentQuery();
        if (!query.trim() && !image_id) {
            resultsEl.innerHTML = "";
            statusEl.innerHTML = `<div class="status-banner info">Type a query to search.</div>`;
            return;
        }
        statusEl.innerHTML = "";

        const use = Object.fromEntries(order.map((key) => [key, document.getElementById(`${prefix}-use-${key}`).checked]));
        if (!order.some((key) => use[key])) {
            resultsEl.innerHTML = `<div class="status-banner info">Check at least one search option in the sidebar.</div>`;
            return;
        }

        const topK = parseInt(document.getElementById("top-k").value, 10) || 200;
        const body = {
            query: image_id ? null : query,
            image_id,
            top_k: topK,
            ...scopeFilters(),
            legs: { ...use },  // only this panel's legs; the API defaults the rest
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

        let exportLeg = null;
        for (const key of EXPORT_LEG_PRIORITY) {
            if (!order.includes(key)) continue;
            const leg = data[key];
            if (use[key] && leg && !leg.skipped) { exportLeg = leg; break; }
        }
        resetExportCandidates(exportLeg ? exportLeg.results : []);

        for (const key of order) {
            if (use[key]) section(resultsEl, labels[key], data[key]);
        }
    }

    return { mount, run };
}
