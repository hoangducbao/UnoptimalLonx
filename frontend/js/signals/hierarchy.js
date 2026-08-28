// frontend/js/signals/hierarchy.js -- Hierarchy Search: bespoke rendering
// (not renderGrid), same as ui/app.py:2112-2179 -- a video-grouped,
// drilled-down result set isn't a plain ranked list.

import { searchHierarchy, expandHierarchy } from "../api.js";
import { renderThumb, renderActions } from "../render.js";
import { currentQuery } from "../query-input.js";
import { hierExtraG, resetExportCandidates, scopeFilters } from "../state.js";

const topGWrap = document.getElementById("top-g-wrap");
const groupByRow = document.getElementById("group-by-row");

// video_id -> {best, step1_frames, seed_n} -- cached so the seed-picker and
// "Expand" button can re-drill without re-running Step 1's base search,
// same as ui/app.py re-reading `groups[vid]` fresh each rerun rather than
// keeping a previously-drilled list (see backend/routes/hierarchy.py's
// module docstring).
let groupsCache = new Map();

export function mount(controlsEl) {
    controlsEl.innerHTML = `<div class="thumb-caption muted">SigLIP2 frame search, grouped by video, drilled down to Top-G frames/video. No leg choice -- text or picture query, SigLIP2 only.</div>`;
    topGWrap.style.display = "block";
    groupByRow.style.display = "none"; // Hierarchy is always grouped by video by construction
}

export function unmount() {
    topGWrap.style.display = "none";
    groupByRow.style.display = "flex";
}

async function renderGroupFrames(vid, container) {
    const g = groupsCache.get(vid);
    const grid = container.querySelector(".hier-frame-grid");
    grid.innerHTML = "";
    // Per-frame actions, same as render.js's grouped renderGrid() -- the
    // group only controls layout, each frame still needs its own targets.
    for (const r of g.drilled_frames) {
        const cell = renderThumb(r);
        cell.append(renderActions(r.video_id, r.n));
        grid.append(cell);
    }
}

async function redrill(vid, container) {
    const g = groupsCache.get(vid);
    const topK = parseInt(document.getElementById("top-k").value, 10) || 30;
    const topG = (parseInt(document.getElementById("top-g").value, 10) || 5) + (hierExtraG.get(vid) || 0);
    const data = await expandHierarchy({
        video_id: vid, step1_frames: g.step1_frames, top_g: topG, seed_n: g.seed_n, top_k: topK,
    });
    g.drilled_frames = data.frames;
    await renderGroupFrames(vid, container);
}

function renderGroup(container, vid) {
    const g = groupsCache.get(vid);
    container.innerHTML = "";

    const header = document.createElement("div");
    header.className = "group-header";
    header.innerHTML = `<b>${vid}</b> · best rank ${g.best_rank} · ${g.score_label}=${g.best_score_val.toFixed(4)} · ${g.step1_frames.length} frame(s) from Step 1`;
    container.append(header);

    // Expand stays video-level (it pulls more frames for this one video --
    // there's no per-frame equivalent), separate from the per-frame
    // show-more/play/copy/export buttons now on each drilled frame below.
    const expandBtn = document.createElement("button");
    expandBtn.className = "btn";
    expandBtn.title = "Pull in 10 more frames from this video";
    expandBtn.textContent = "⬇ Expand";
    expandBtn.style.margin = "0.3rem 0";
    expandBtn.onclick = () => {
        hierExtraG.set(vid, (hierExtraG.get(vid) || 0) + 10);
        redrill(vid, container);
    };
    container.append(expandBtn);

    const seedSelect = document.createElement("select");
    seedSelect.style.cssText = "width:100%;margin:0.3rem 0;padding:0.35rem;border:1px solid var(--border);border-radius:6px;";
    for (const n of g.seed_options) {
        const opt = document.createElement("option");
        opt.value = n;
        opt.textContent = `Seed: frame ${n}` + (n === g.top1_n ? " (top-1)" : "");
        if (n === g.seed_n) opt.selected = true;
        seedSelect.append(opt);
    }
    seedSelect.onchange = () => {
        g.seed_n = parseInt(seedSelect.value, 10);
        redrill(vid, container);
    };
    container.append(seedSelect);

    const grid = document.createElement("div");
    grid.className = "grid hier-frame-grid";
    container.append(grid);

    const hr = document.createElement("hr");
    hr.className = "divider";
    container.append(hr);

    renderGroupFrames(vid, container);
}

export async function run(resultsEl, statusEl) {
    const { query, image_id } = currentQuery();
    if (!query.trim() && !image_id) {
        resultsEl.innerHTML = "";
        statusEl.innerHTML = `<div class="status-banner info">Type a query, or paste an image, to search.</div>`;
        return;
    }
    statusEl.innerHTML = "";

    const topK = parseInt(document.getElementById("top-k").value, 10) || 30;
    const topG = parseInt(document.getElementById("top-g").value, 10) || 5;
    const body = { query: image_id ? null : query, image_id, top_k: topK, top_g: topG, ...scopeFilters() };

    resultsEl.innerHTML = `<div class="status-banner info">Searching…</div>`;
    let data;
    try {
        data = await searchHierarchy(body);
    } catch (e) {
        resultsEl.innerHTML = `<div class="status-banner error">${e.message}</div>`;
        return;
    }
    resultsEl.innerHTML = "";

    if (!data.groups.length) {
        resultsEl.innerHTML = `<div class="status-banner info">No results.</div>`;
        return;
    }
    // Flatten to one candidate per video (its Step 1 top-1 frame, the same
    // frame renderActions()'s export (★) button targets below) -- Hierarchy's
    // grouped/drilled-down shape isn't a plain ranked list, but the export
    // popup's "similars" preview needs one.
    resetExportCandidates(data.groups.map((g) => ({
        video_id: g.video_id, n: g.step1_frames[0].n, rank: g.best_rank,
        score_label: g.score_label, score_val: g.best_score_val, text: null,
    })));

    groupsCache = new Map();
    const h = document.createElement("h2");
    h.textContent = "Hierarchy Search";
    resultsEl.append(h);

    for (const g of data.groups) {
        groupsCache.set(g.video_id, {
            best_rank: g.best_rank, score_label: g.score_label, best_score_val: g.best_score_val,
            best_n: g.step1_frames[0].n, step1_frames: g.step1_frames, seed_options: g.seed_options,
            top1_n: g.top1_n, seed_n: g.top1_n, drilled_frames: g.drilled_frames,
        });
        const groupEl = document.createElement("div");
        resultsEl.append(groupEl);
        renderGroup(groupEl, g.video_id);
    }
}
