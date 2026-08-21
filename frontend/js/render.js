// frontend/js/render.js -- renderThumb/renderActions/renderGrid, a direct
// port of ui/app.py:1461-1567 (render_thumb/render_actions/render_grid).
// Every signal calls renderGrid() with the same {video_id, n, rank,
// score_label, score_val, text, thumbnail_url} shape backend/common.py's
// df_to_results() produces -- one renderer for every signal, same as the
// original's "one render_grid() serves every signal" design.

import { openNeighborsDialog, openPlaybackDialog } from "./dialogs.js";
import { copyCollectionOnly, copyToScope } from "./state.js";

function videoLotStr(videoId) {
    const m = /^L(\d+)/i.exec(videoId);
    return m ? `L${m[1]}` : videoId;
}

function showFullText() {
    return document.getElementById("show-full-text").checked;
}

export function renderThumb(r) {
    const cell = document.createElement("div");
    cell.className = "thumb-cell";

    cell.innerHTML = r.thumbnail_url
        ? `<div class="thumb-wrap"><img src="${r.thumbnail_url}" loading="lazy"></div>`
        : `<div class="thumb-missing">(image not found)</div>`;

    const cap1 = document.createElement("div");
    cap1.className = "thumb-caption";
    cap1.innerHTML = `<b>${r.video_id}</b> · frame ${r.n}`;
    cell.append(cap1);

    const cap2 = document.createElement("div");
    cap2.className = "thumb-caption muted";
    cap2.textContent = `rank ${r.rank} · ${r.score_label}=${r.score_val.toFixed(4)}`;
    cell.append(cap2);

    if (r.text) {
        const textEl = document.createElement("div");
        textEl.className = "thumb-text";
        textEl.textContent = showFullText() || r.text.length <= 140
            ? r.text
            : r.text.slice(0, 140) + "…";
        cell.append(textEl);
    }

    return cell;
}

export function renderActions(videoId, centerN, { collectionOnly = false } = {}) {
    const row = document.createElement("div");
    row.className = "thumb-actions";

    const moreBtn = document.createElement("button");
    moreBtn.className = "icon-btn";
    moreBtn.title = "Show more";
    moreBtn.textContent = "⋯";
    moreBtn.onclick = () => openNeighborsDialog(videoId, centerN);

    const playBtn = document.createElement("button");
    playBtn.className = "icon-btn";
    playBtn.title = "Play video";
    playBtn.textContent = "▶";
    playBtn.onclick = () => openPlaybackDialog(videoId, centerN);

    const copyBtn = document.createElement("button");
    copyBtn.className = "icon-btn";
    copyBtn.title = collectionOnly ? "Copy collection id" : "Copy video id";
    copyBtn.textContent = "⧉";
    copyBtn.onclick = () => collectionOnly ? copyCollectionOnly(videoId) : copyToScope(videoId);

    row.append(moreBtn, playBtn, copyBtn);
    return row;
}

/**
 * groupMode: null (ungrouped), "video" (group frames by video_id), or
 * "collection" (group by lot -- Summary mode).
 */
export function renderGrid(container, results, groupMode = null) {
    container.innerHTML = "";
    if (!results || results.length === 0) {
        container.innerHTML = `<div class="status-banner info">No results.</div>`;
        return;
    }

    if (!groupMode) {
        const grid = document.createElement("div");
        grid.className = "grid";
        for (const r of results) {
            const cell = renderThumb(r);
            cell.append(renderActions(r.video_id, r.n));
            grid.append(cell);
        }
        container.append(grid);
        return;
    }

    const groupKeyFn = groupMode === "collection" ? videoLotStr : (vid) => vid;
    const unit = groupMode === "collection" ? "video(s)" : "frame(s)";
    const groups = new Map();
    for (const r of results) {
        const key = groupKeyFn(r.video_id);
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(r);
    }
    // results already arrive rank-sorted, so a group's first occurrence is
    // its best (lowest-rank) member -- that also decides display order.
    const ordered = [...groups.entries()].sort((a, b) => a[1][0].rank - b[1][0].rank);

    for (const [groupKey, frames] of ordered) {
        const best = frames[0];
        const header = document.createElement("div");
        header.className = "group-header";
        header.innerHTML = `<b>${groupKey}</b> · best rank ${best.rank} · ${best.score_label}=${best.score_val.toFixed(4)} · ${frames.length} ${unit}`;
        container.append(header);

        const grid = document.createElement("div");
        grid.className = "grid";
        for (const r of frames) grid.append(renderThumb(r));
        container.append(grid);

        container.append(renderActions(best.video_id, best.n, { collectionOnly: groupMode === "collection" }));

        const hr = document.createElement("hr");
        hr.className = "divider";
        container.append(hr);
    }
}
