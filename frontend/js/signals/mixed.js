// frontend/js/signals/mixed.js -- Mixed signal panel. Ports ui/app.py's
// standalone Mixed mode (ui/app.py:1798, 1799-1800, 2089-2110): a "Change
// weights" button + a caption of the current weights, reading/writing the
// one shared mixedConfig every TRAKE-row-set-to-Mixed will also use later.

import { searchMixed } from "../api.js";
import { renderGrid } from "../render.js";
import { currentQuery } from "../query-input.js";
import { openWeightsDialog } from "../dialogs.js";
import { mixedConfig, resetExportCandidates, scopeFilters } from "../state.js";

let controlsElRef = null;
let runRef = () => {};

function weightsCaption() {
    const w = mixedConfig.weights;
    return `Weights — Keyframe ${w.Keyframe} · ASR ${w.ASR} · Caption ${w.Caption} · OCR ${w.OCR}`;
}

export function mount(controlsEl) {
    controlsElRef = controlsEl;
    controlsEl.innerHTML = `
      <button class="btn" id="mixed-weights-btn" title="Change weights">⚙ Change weights</button>
      <div class="thumb-caption muted" id="mixed-weights-caption" style="margin-top:0.4rem;"></div>`;
    controlsEl.querySelector("#mixed-weights-caption").textContent = weightsCaption();
    controlsEl.querySelector("#mixed-weights-btn").onclick = () => {
        openWeightsDialog(() => {
            controlsEl.querySelector("#mixed-weights-caption").textContent = weightsCaption();
            runRef();
        });
    };
}

function groupMode() {
    return document.getElementById("group-by-video").checked ? "video" : null;
}

export async function run(resultsEl, statusEl) {
    runRef = () => run(resultsEl, statusEl);
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
        weights: mixedConfig.weights,
        legs: mixedConfig.legs,
    };

    resultsEl.innerHTML = `<div class="status-banner info">Searching…</div>`;
    let data;
    try {
        data = await searchMixed(body);
    } catch (e) {
        resultsEl.innerHTML = `<div class="status-banner error">${e.message}</div>`;
        return;
    }
    resultsEl.innerHTML = "";

    if (data.empty) {
        resultsEl.innerHTML = `<div class="status-banner info">Every signal weight is 0 — open <b>Change weights</b> and enable at least one.</div>`;
        return;
    }
    resetExportCandidates(data.results);

    const h = document.createElement("h2");
    h.textContent = "Mixed (weighted RRF)";
    resultsEl.append(h);
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
