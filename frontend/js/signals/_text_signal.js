// frontend/js/signals/_text_signal.js -- ASR/Caption/Summary share one
// shape (SigLIP2 leg + fuzzy leg + RRF, ui/app.py:2000-2050, 2063-2087),
// differing only in labels, the API call, (Summary) group-by-collection
// instead of group-by-video, and (per-instance) checkbox order/defaults.
// This factory avoids repeating that shape three times; OCR is the
// one-leg exception and gets its own file.

import { renderGrid } from "../render.js";
import { currentQuery } from "../query-input.js";
import { resetExportCandidates, scopeFilters } from "../state.js";

const LEG_KEYS = ["siglip", "fuzzy", "rrf"];
// Export candidates come from the single "best" enabled leg, not every leg
// rendered on screen -- RRF (fused) is preferred when enabled, since it's
// the strongest ranking available, falling back to whichever plain leg is
// on when RRF isn't.
const EXPORT_LEG_PRIORITY = ["rrf", "siglip", "fuzzy"];

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
            ...scopeFilters(),
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
        let exportLeg = null;
        for (const key of EXPORT_LEG_PRIORITY) {
            const leg = data[dataKey[key]];
            if (use[key] && leg && !leg.skipped) { exportLeg = leg; break; }
        }
        resetExportCandidates(exportLeg ? exportLeg.results : []);

        for (const key of order) {
            if (use[key]) section(resultsEl, labels[key], data[dataKey[key]]);
        }
    }

    return { mount, run };
}
