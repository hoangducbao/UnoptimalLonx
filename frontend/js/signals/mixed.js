// frontend/js/signals/mixed.js -- Mixed signal panel: many independent
// sub-queries, each with its own text + single signal (no nesting, no
// image sub-queries), combined via weighted RRF (weight 0-3 per
// sub-query). UI structure mirrors signals/trake.js's add/remove-row list,
// minus the context row and the temporal/ordering pieces.

import { searchMixed } from "../api.js";
import { renderGrid } from "../render.js";
import { signalSelectHtml } from "../signal-select.js";
import { mixedQueryState, resetExportCandidates, scopeFilters, MIXED_QUERY_SIGNALS } from "../state.js";

const mixedSection = document.getElementById("mixed-query-section");
const standardSection = document.getElementById("standard-query-section");

let runRef = () => {};

function bindEnterSubmit(textarea) {
    textarea.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.isComposing) {
            e.preventDefault();
            runRef();
        }
    });
}

function renderInputs() {
    mixedSection.innerHTML = `
        <div class="thumb-caption muted" style="margin:0.5rem 0 0.25rem;">Sub-queries -- combined by weighted RRF</div>
        <div id="mixed-subquery-rows"></div>
        <button class="btn" id="mixed-add-subquery" style="width:100%;margin-top:0.4rem;">+ Add sub-query</button>`;

    const rowsEl = mixedSection.querySelector("#mixed-subquery-rows");
    mixedQueryState.queries.forEach((q, i) => {
        const row = document.createElement("div");
        row.style.cssText = "margin:0.5rem 0;padding:0.5rem;border:1px solid var(--border);border-radius:6px;";
        row.innerHTML = `<div class="thumb-caption muted" style="margin-bottom:0.25rem;">Sub-query ${i + 1}</div>
            <textarea placeholder="sub-query text" style="height:56px;"></textarea>
            <div style="display:flex;gap:0.4rem;align-items:center;margin-top:0.3rem;">
              <div style="flex:1;">${signalSelectHtml(`mixed-signal-${q.id}`, q.signal, MIXED_QUERY_SIGNALS)}</div>
              <button class="icon-btn" title="Remove sub-query" ${mixedQueryState.queries.length <= 1 ? "disabled" : ""}>✕</button>
            </div>
            <div style="display:flex;align-items:center;gap:0.5rem;margin-top:0.4rem;">
              <label class="thumb-caption muted" style="flex:0 0 auto;margin:0;">Weight (<span class="mixed-weight-val">${q.weight}</span>)</label>
              <input type="range" min="0" max="3" step="1" style="flex:1;">
            </div>`;
        const ta = row.querySelector("textarea");
        ta.value = q.text;
        ta.oninput = () => { q.text = ta.value; };
        bindEnterSubmit(ta);

        const sel = row.querySelector("select");
        sel.onchange = () => { q.signal = sel.value; runRef(); };

        const slider = row.querySelector("input[type=range]");
        slider.value = q.weight;
        const weightLabel = row.querySelector(".mixed-weight-val");
        slider.oninput = () => {
            q.weight = parseInt(slider.value, 10);
            weightLabel.textContent = q.weight;
        };
        slider.onchange = () => runRef();

        const removeBtn = row.querySelector(".icon-btn");
        removeBtn.onclick = () => {
            mixedQueryState.queries = mixedQueryState.queries.filter((x) => x.id !== q.id);
            renderInputs();
            runRef();
        };
        rowsEl.append(row);
    });

    mixedSection.querySelector("#mixed-add-subquery").onclick = () => {
        mixedQueryState.queries.push({ id: mixedQueryState.nextId++, text: "", signal: "Keyframe", weight: 1 });
        renderInputs();
    };
}

export function mount(controlsEl) {
    controlsEl.innerHTML = `
      <div class="checkbox-row">
        <input type="checkbox" id="mixed-show-transcript">
        <label for="mixed-show-transcript" style="margin:0;">Show transcript</label>
      </div>`;
    standardSection.style.display = "none";
    mixedSection.style.display = "block";
    renderInputs();
}

export function unmount() {
    standardSection.style.display = "block";
    mixedSection.style.display = "none";
}

function groupMode() {
    return document.getElementById("group-by-video").checked ? "video" : null;
}

export async function run(resultsEl, statusEl) {
    runRef = () => run(resultsEl, statusEl);
    const active = mixedQueryState.queries.filter((q) => q.text.trim() && q.weight > 0);
    if (active.length === 0) {
        resultsEl.innerHTML = "";
        statusEl.innerHTML = `<div class="status-banner info">Add at least one sub-query with text and a weight above 0.</div>`;
        return;
    }
    statusEl.innerHTML = "";

    const topK = parseInt(document.getElementById("top-k").value, 10) || 100;
    const body = {
        queries: active.map((q) => ({ text: q.text, signal: q.signal, weight: q.weight })),
        top_k: topK,
        ...scopeFilters(),
        show_transcript: document.getElementById("mixed-show-transcript").checked,
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
        resultsEl.innerHTML = `<div class="status-banner info">Add at least one sub-query with text and a weight above 0.</div>`;
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
