// frontend/js/signals/trake.js -- TRAKE signal panel. Ports ui/app.py's
// TRAKE sidebar (context row E0 + dynamic event rows E1..En with add/
// remove, ui/app.py:1801-1893) and result cards (ui/app.py:2181-2262).

import { searchTrake } from "../api.js";
import { openTrakePlaybackDialog, openWeightsDialog } from "../dialogs.js";
import { openExportDialog } from "../export-dialog.js";
import { copyToScope, mixedConfig, resetExportCandidates, scopeFilters, trakeState, TRAKE_EVENT_SIGNALS } from "../state.js";

const trakeSection = document.getElementById("trake-query-section");
const standardSection = document.getElementById("standard-query-section");
const groupByRow = document.getElementById("group-by-row");
const topVWrap = document.getElementById("top-v-wrap");

let runRef = () => {};

function signalSelectHtml(id, current) {
    const opts = TRAKE_EVENT_SIGNALS.map((s) => `<option value="${s}"${s === current ? " selected" : ""}>${s}</option>`).join("");
    return `<select id="${id}" style="width:100%;padding:0.35rem;border:1px solid var(--border);border-radius:6px;">${opts}</select>`;
}

function bindEnterSubmit(textarea) {
    // Trivial now -- ui/app.py's version needed a MutationObserver +
    // singleton guard purely to survive Streamlit re-emitting the same
    // <script> on every rerun (ui/app.py:1865-1893); a hand-written page
    // just attaches the listener once, when the row is created.
    textarea.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.isComposing) {
            e.preventDefault();
            runRef();
        }
    });
}

function renderInputs() {
    trakeSection.innerHTML = "";

    // Context row (E0, optional).
    const ctxWrap = document.createElement("div");
    ctxWrap.innerHTML = `<div class="thumb-caption muted" style="margin:0.5rem 0 0.25rem;">Context — E0 (optional, boosts matching videos)</div>
        <textarea id="trake-ctx-text" placeholder="optional context query" style="height:56px;"></textarea>
        <div style="display:flex;gap:0.4rem;align-items:center;margin:0.3rem 0;">
          <div style="flex:1;">${signalSelectHtml("trake-ctx-signal", trakeState.context.signal)}</div>
          <button class="icon-btn" id="trake-ctx-weights" title="Change weights" style="display:${trakeState.context.signal === "Mixed" ? "flex" : "none"};">⚙</button>
        </div>
        <hr class="divider">
        <div class="thumb-caption muted" style="margin-bottom:0.25rem;">Events, in required order</div>
        <div id="trake-event-rows"></div>
        <button class="btn" id="trake-add-event" style="width:100%;margin-top:0.4rem;">+ Add event</button>`;
    trakeSection.append(ctxWrap);

    const ctxText = trakeSection.querySelector("#trake-ctx-text");
    ctxText.value = trakeState.context.text;
    ctxText.oninput = () => { trakeState.context.text = ctxText.value; };
    bindEnterSubmit(ctxText);

    const ctxSignal = trakeSection.querySelector("#trake-ctx-signal");
    const ctxWeightsBtn = trakeSection.querySelector("#trake-ctx-weights");
    ctxSignal.onchange = () => {
        trakeState.context.signal = ctxSignal.value;
        ctxWeightsBtn.style.display = ctxSignal.value === "Mixed" ? "flex" : "none";
        runRef();
    };
    ctxWeightsBtn.onclick = () => openWeightsDialog(runRef);

    const rowsEl = trakeSection.querySelector("#trake-event-rows");
    trakeState.events.forEach((ev, i) => {
        const row = document.createElement("div");
        row.style.cssText = "margin:0.5rem 0;padding:0.5rem;border:1px solid var(--border);border-radius:6px;";
        row.innerHTML = `<div class="thumb-caption muted" style="margin-bottom:0.25rem;">Event ${i + 1}</div>
            <textarea placeholder="E${i + 1} query text" style="height:56px;"></textarea>
            <div style="display:flex;gap:0.4rem;align-items:center;margin-top:0.3rem;">
              <div style="flex:1;">${signalSelectHtml(`trake-ev-signal-${ev.id}`, ev.signal)}</div>
              <button class="icon-btn" title="Change weights" style="display:${ev.signal === "Mixed" ? "flex" : "none"};">⚙</button>
              <button class="icon-btn" title="Remove event" ${trakeState.events.length <= 1 ? "disabled" : ""}>✕</button>
            </div>`;
        const ta = row.querySelector("textarea");
        ta.value = ev.text;
        ta.oninput = () => { ev.text = ta.value; };
        bindEnterSubmit(ta);

        const sel = row.querySelector("select");
        const [weightsBtn, removeBtn] = row.querySelectorAll(".icon-btn");
        sel.onchange = () => {
            ev.signal = sel.value;
            weightsBtn.style.display = sel.value === "Mixed" ? "flex" : "none";
            runRef();
        };
        weightsBtn.onclick = () => openWeightsDialog(runRef);
        removeBtn.onclick = () => {
            trakeState.events = trakeState.events.filter((e) => e.id !== ev.id);
            renderInputs();
            runRef();
        };
        rowsEl.append(row);
    });

    trakeSection.querySelector("#trake-add-event").onclick = () => {
        trakeState.events.push({ id: trakeState.nextId++, text: "", signal: "Keyframe" });
        renderInputs();
    };
}

export function mount(controlsEl) {
    controlsEl.innerHTML = "";
    standardSection.style.display = "none";
    trakeSection.style.display = "block";
    groupByRow.style.display = "none";
    topVWrap.style.display = "block";
    renderInputs();
}

export function unmount() {
    standardSection.style.display = "block";
    trakeSection.style.display = "none";
    groupByRow.style.display = "flex";
    topVWrap.style.display = "none";
}

function renderCandidate(container, c) {
    const nMatched = c.events.filter((e) => e.matched).length;
    const header = document.createElement("div");
    header.className = "group-header";
    header.innerHTML = `<b>${c.video_id}</b> · video_score=${c.video_score.toFixed(4)} · coverage ${nMatched}/${c.events.length}`;
    container.append(header);

    const nDisplay = Math.max(2, c.events.length);
    const grid = document.createElement("div");
    grid.className = "grid";
    grid.style.gridTemplateColumns = `repeat(${nDisplay}, 1fr)`;
    const thumbClass = c.events.length >= 3 ? "thumb-wrap" : "thumb-wrap thumb-wrap-static";

    for (const e of c.events) {
        const cell = document.createElement("div");
        cell.className = "thumb-cell";
        if (e.matched) {
            cell.innerHTML = `<div class="${thumbClass}"><img src="${e.thumbnail_url}"></div>
                <div class="thumb-caption"><b>${e.label}</b> · frame ${e.n}</div>
                <div class="thumb-caption muted">${e.score_label}=${e.score_val.toFixed(4)}</div>`;
        } else {
            cell.innerHTML = `<div class="thumb-caption"><b>${e.label}</b></div><div class="thumb-caption muted">no match</div>`;
        }
        grid.append(cell);
    }
    container.append(grid);

    // Single play-icon action per video: acts as both playback and "copy
    // scope" (ui/app.py:2250-2261) -- per-event thumbnails above are
    // display-only, no own actions.
    const playBtn = document.createElement("button");
    playBtn.className = "icon-btn";
    playBtn.title = "Video playback";
    playBtn.textContent = "▶";
    playBtn.onclick = () => {
        copyToScope(c.video_id);
        openTrakePlaybackDialog(c.video_id, c.events);
    };
    container.append(playBtn);

    const exportBtn = document.createElement("button");
    exportBtn.className = "icon-btn";
    exportBtn.title = "Export as AIC submission CSV";
    exportBtn.textContent = "★";
    exportBtn.onclick = () => openExportDialog({ kind: "trake", candidate: c });
    container.append(exportBtn);

    const hr = document.createElement("hr");
    hr.className = "divider";
    container.append(hr);
}

export async function run(resultsEl, statusEl) {
    runRef = () => run(resultsEl, statusEl);
    const texts = trakeState.events.map((e) => e.text.trim());
    if (trakeState.events.length < 1 || !texts.every(Boolean)) {
        resultsEl.innerHTML = "";
        statusEl.innerHTML = `<div class="status-banner info">Fill in every event's query text to search (minimum 1 event).</div>`;
        return;
    }
    statusEl.innerHTML = "";

    const topK = parseInt(document.getElementById("top-k").value, 10) || 30;
    const topV = parseInt(document.getElementById("top-v").value, 10) || 10;
    const body = {
        context: trakeState.context.text.trim() ? trakeState.context : null,
        events: trakeState.events.map((e) => ({ text: e.text, signal: e.signal })),
        top_k: topK,
        top_v: topV,
        ...scopeFilters(),
        mixed_weights: mixedConfig.weights,
        mixed_legs: mixedConfig.legs,
    };

    resultsEl.innerHTML = `<div class="status-banner info">Searching…</div>`;
    let data;
    try {
        data = await searchTrake(body);
    } catch (e) {
        resultsEl.innerHTML = `<div class="status-banner error">${e.message}</div>`;
        return;
    }
    resultsEl.innerHTML = "";
    resetExportCandidates(data.candidates || []);

    if (data.message) {
        resultsEl.innerHTML = `<div class="status-banner info">${data.message}</div>`;
        return;
    }

    const h = document.createElement("h2");
    h.textContent = "TRAKE";
    resultsEl.append(h);
    for (const c of data.candidates) renderCandidate(resultsEl, c);
}
