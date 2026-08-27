// frontend/js/dialogs.js -- modal dialogs. Phase 1: "Nearby frames" (ports
// ui/app.py's show_neighbors, ui/app.py:1334-1360) and single-frame
// playback (ports frame_playback_dialog, ui/app.py:1363-1376). TRAKE's
// marker-bar playback and the Mixed change-weights dialog land in their
// own phases, same file.

import { getNeighbors, getPlayback } from "./api.js";
// export-dialog.js now just opens the Export CSV tab (frontend/export.html,
// see its own header) rather than a modal built from openDialog() here --
// no circular import with this file any more.
import { openExportDialog } from "./export-dialog.js";
import {
    getNeighborExtra, MIXED_DEFAULT_LEGS, MIXED_DEFAULT_WEIGHTS,
    MIXED_LEG_DEFS, MIXED_SIGNAL_NAMES, mixedConfig, saveMixedConfig,
} from "./state.js";

const root = document.getElementById("dialog-root");

// `title` may be falsy (null/"") to omit the heading entirely -- used by
// the playback dialogs, which put video_id/frame/timer info beside the
// video instead of needing a heading above it. The close button is
// absolutely positioned (not floated) specifically so it works the same
// way whether or not a title/h3 is present.
export function openDialog(title, bodyEl, { wide = false } = {}) {
    const overlay = document.createElement("div");
    overlay.className = "dialog-overlay";
    const box = document.createElement("div");
    box.className = "dialog-box" + (wide ? " wide" : "");
    const closeBtn = document.createElement("button");
    closeBtn.className = "dialog-close";
    closeBtn.textContent = "✕";
    closeBtn.onclick = () => overlay.remove();
    overlay.onclick = (e) => { if (e.target === overlay) overlay.remove(); };

    box.append(closeBtn);
    if (title) {
        const h3 = document.createElement("h3");
        h3.textContent = title;
        box.append(h3);
    }
    box.append(bodyEl);
    overlay.append(box);
    root.append(overlay);
    return { overlay, box };
}

export async function openNeighborsDialog(videoId, centerN) {
    const body = document.createElement("div");
    body.innerHTML = `<div class="thumb-caption" style="margin-bottom:0.5rem;">${videoId} — around frame ${centerN}</div>
        <button class="btn" id="nbr-up" style="width:100%;margin-bottom:0.5rem;">▲ 10 earlier</button>
        <div class="grid nbr-grid" id="nbr-grid" style="grid-template-columns:repeat(5,1fr);"></div>
        <button class="btn" id="nbr-down" style="width:100%;margin-top:0.5rem;">▼ 10 later</button>`;
    const { box } = openDialog("Nearby frames", body, { wide: true });

    async function refresh() {
        const extra = getNeighborExtra(videoId, centerN);
        const data = await getNeighbors(videoId, centerN, extra.before, extra.after);
        const grid = box.querySelector("#nbr-grid");
        grid.innerHTML = "";
        for (const f of data.frames) {
            const cell = document.createElement("div");
            cell.className = "thumb-cell";
            // thumb-wrap-static suppresses the normal hover-zoom (same
            // class TRAKE's low-count cards use) -- distracting in this
            // tightly packed nearby-frames grid.
            cell.innerHTML = f.exists
                ? `<div class="thumb-wrap thumb-wrap-static"><img src="${f.thumbnail_url}"></div>`
                : `<div class="thumb-missing">(missing)</div>`;
            const cap = document.createElement("div");
            cap.className = "thumb-caption";
            cap.innerHTML = f.is_center ? `<b>${f.n}</b>` : String(f.n);
            cell.append(cap);
            if (f.exists) {
                // Reuses .export-add-btn's CSS (top-right overlay corner,
                // same as the export screen's own preview cards) purely for
                // position/sizing -- unrelated to that button's add/replace
                // behavior elsewhere.
                const exportBtn = document.createElement("button");
                exportBtn.className = "icon-btn export-add-btn";
                exportBtn.title = "Export as AIC submission CSV";
                exportBtn.textContent = "★";
                exportBtn.onclick = () => openExportDialog({ kind: "flat", video_id: videoId, n: f.n });
                cell.append(exportBtn);
            }
            grid.append(cell);
        }
    }

    box.querySelector("#nbr-up").onclick = () => {
        getNeighborExtra(videoId, centerN).before += 10;
        refresh();
    };
    box.querySelector("#nbr-down").onclick = () => {
        getNeighborExtra(videoId, centerN).after += 10;
        refresh();
    };

    await refresh();
}

function fmtTime(t) {
    const mm = String(Math.floor(t / 60)).padStart(2, "0");
    const ss = (t % 60).toFixed(2).padStart(5, "0");
    return `${mm}:${ss}`;
}

export async function openPlaybackDialog(videoId, n) {
    const body = document.createElement("div");
    body.innerHTML = `<div class="playback-layout">
        <div class="playback-main" id="playback-video-wrap">Loading…</div>
        <div class="playback-info">
          <div class="thumb-caption">${videoId} — frame ${n}</div>
          <div id="playback-timer" class="playback-timer">--:-- · frame --</div>
          <button class="btn" id="playback-export-btn" style="margin-top:0.5rem;" title="Export the exact frame currently playing">★ Export this frame</button>
        </div>
      </div>`;
    const { box } = openDialog(null, body);

    try {
        const data = await getPlayback(videoId, n);
        const wrap = box.querySelector("#playback-video-wrap");
        wrap.innerHTML = "";
        const video = document.createElement("video");
        video.src = data.video_url + `#t=${data.start_time}`;
        video.controls = true;
        video.autoplay = false;
        wrap.append(video);

        // Live realtime frame timer -- same fps*currentTime readout TRAKE's
        // marker-bar playback uses (ui/app.py:1440-1442's `timeupdate`
        // handler), just without the marker bar since there's only one
        // frame here.
        const timer = box.querySelector("#playback-timer");
        video.addEventListener("timeupdate", () => {
            timer.textContent = `${fmtTime(video.currentTime)} · frame ${Math.round(video.currentTime * data.fps)}`;
        });

        // Exports whatever real frame the video is currently at (paused or
        // not), computed fresh at click time -- not read off the timer's
        // text, which is just a display of the same arithmetic. No
        // keyframe n involved at all, so this always opens as a TRAKE
        // export (see export-ui.js's {kind:"frame"} handling).
        box.querySelector("#playback-export-btn").onclick = () => {
            openExportDialog({ kind: "frame", video_id: videoId, frame_idx: Math.round(video.currentTime * data.fps) });
        };
    } catch (e) {
        box.querySelector("#playback-video-wrap").innerHTML =
            `<div class="status-banner error">${e.message}</div>`;
    }
}

// "Change weights" dialog -- ports ui/app.py's change_weights_dialog
// (ui/app.py:1286-1312). Edits a staged copy so Cancel discards changes;
// Save commits into the one shared mixedConfig (state.js) and persists it,
// same as ui/app.py committing into st.session_state.mixed_weights/legs.
// `onSave` lets the caller (standalone Mixed mode, or a TRAKE row in a
// later phase) re-run its search after a Save.
export function openWeightsDialog(onSave) {
    const staged = {
        weights: { ...mixedConfig.weights },
        legs: { ...mixedConfig.legs },
    };

    const body = document.createElement("div");
    body.innerHTML = `<div class="thumb-caption muted" style="margin-bottom:0.5rem;">Weight per signal (0 = off) with that signal's legs alongside</div>
        <div id="weights-rows"></div>
        <hr class="divider">
        <div style="display:flex;gap:0.5rem;">
          <button class="btn" id="weights-default">Default</button>
          <button class="btn" id="weights-cancel">Cancel</button>
          <button class="btn btn-primary" id="weights-save">Save</button>
        </div>`;
    const { overlay, box } = openDialog("Change weights", body);
    const rows = box.querySelector("#weights-rows");

    function renderRows() {
        rows.innerHTML = "";
        for (const name of MIXED_SIGNAL_NAMES) {
            const row = document.createElement("div");
            row.style.cssText = "display:flex;align-items:center;gap:1rem;margin:0.5rem 0;";
            const label = document.createElement("label");
            label.style.cssText = "flex:0 0 90px;margin:0;";
            label.textContent = `${name} (${staged.weights[name]})`;
            const slider = document.createElement("input");
            slider.type = "range";
            slider.min = "0"; slider.max = "3"; slider.step = "1";
            slider.value = staged.weights[name];
            slider.style.flex = "1 1 auto";
            slider.oninput = () => {
                staged.weights[name] = parseInt(slider.value, 10);
                label.textContent = `${name} (${staged.weights[name]})`;
            };
            row.append(label, slider);

            const legsBox = document.createElement("div");
            legsBox.style.cssText = "flex:0 0 200px;display:flex;flex-direction:column;gap:0.15rem;";
            if (MIXED_LEG_DEFS[name]) {
                for (const [legKey, legLabel] of MIXED_LEG_DEFS[name]) {
                    const cbRow = document.createElement("label");
                    cbRow.style.cssText = "display:flex;align-items:center;gap:0.3rem;font-size:0.85rem;margin:0;";
                    const cb = document.createElement("input");
                    cb.type = "checkbox";
                    cb.checked = staged.legs[legKey];
                    cb.onchange = () => { staged.legs[legKey] = cb.checked; };
                    cbRow.append(cb, document.createTextNode(legLabel));
                    legsBox.append(cbRow);
                }
            } else {
                legsBox.innerHTML = `<span class="thumb-caption muted">Detailed legs</span>`;
            }
            row.append(legsBox);
            rows.append(row);
        }
    }
    renderRows();

    box.querySelector("#weights-default").onclick = () => {
        staged.weights = { ...MIXED_DEFAULT_WEIGHTS };
        staged.legs = { ...MIXED_DEFAULT_LEGS };
        renderRows();
    };
    box.querySelector("#weights-cancel").onclick = () => overlay.remove();
    box.querySelector("#weights-save").onclick = () => {
        mixedConfig.weights = staged.weights;
        mixedConfig.legs = staged.legs;
        saveMixedConfig();
        overlay.remove();
        if (onSave) onSave();
    };
}

// TRAKE's play-icon action: opens the source video seeked near the first
// matched event, with a click-to-seek marker row for every matched event
// and a live timestamp/frame readout. Ports trake_playback_dialog +
// render_trake_playback_binder's JS (ui/app.py:1379-1492) -- collapsed
// into one function here since a hand-written page has no
// "st.dialog can't run <script>" limitation to work around, and no
// rerun-driven observer/singleton-guard dance needed either.
export async function openTrakePlaybackDialog(videoId, events) {
    const matched = events.filter((e) => e.matched && e.timestamp !== null);
    const gaps = events.filter((e) => !e.matched);

    const body = document.createElement("div");
    body.innerHTML = `<div class="playback-layout">
        <div class="playback-main">
          <div id="trake-video-wrap">Loading…</div>
          <div id="trake-marker-bar"></div>
        </div>
        <div class="playback-info">
          <div class="thumb-caption">${videoId}</div>
          <div id="trake-timer" class="playback-timer">--:-- · frame --</div>
          <button class="btn" id="trake-export-btn" style="margin-top:0.5rem;" title="Export the exact frame currently playing">★ Export this frame</button>
          <div id="trake-gaps"></div>
        </div>
      </div>`;
    const { box } = openDialog(null, body, { wide: true });

    if (gaps.length) {
        const gapsEl = box.querySelector("#trake-gaps");
        gapsEl.innerHTML = `<hr class="divider"><div class="thumb-caption muted" style="margin-bottom:0.4rem;">Coverage gaps — scrub manually between the nearest matched anchors:</div>`;
        for (const e of gaps) {
            const before = matched.filter((m) => m.event_index < e.event_index).at(-1);
            const after = matched.find((m) => m.event_index > e.event_index);
            const lo = before ? `${before.timestamp.toFixed(2)}s (${before.label})` : "start";
            const hi = after ? `${after.timestamp.toFixed(2)}s (${after.label})` : "end";
            const line = document.createElement("div");
            line.className = "thumb-caption";
            line.innerHTML = `${e.label}: no direct match — between <b>${lo}</b> and <b>${hi}</b>`;
            gapsEl.append(line);
        }
    }

    if (!matched.length) {
        box.querySelector("#trake-video-wrap").innerHTML = `<div class="status-banner info">No matched events to seek to.</div>`;
        const exportBtn = box.querySelector("#trake-export-btn");
        exportBtn.disabled = true;
        exportBtn.title = "No video loaded to read a frame from";
        return;
    }

    try {
        const data = await getPlayback(videoId, matched[0].n);
        const wrap = box.querySelector("#trake-video-wrap");
        wrap.innerHTML = "";
        const video = document.createElement("video");
        video.src = data.video_url + `#t=${matched[0].timestamp}`;
        video.controls = true;
        wrap.append(video);

        const bar = box.querySelector("#trake-marker-bar");
        const timer = box.querySelector("#trake-timer");

        function layoutMarkers() {
            if (!video.duration || !isFinite(video.duration)) return;
            bar.innerHTML = "";
            for (const m of matched) {
                const pct = Math.max(0, Math.min(100, (m.timestamp / video.duration) * 100));
                const tick = document.createElement("div");
                tick.className = "trake-marker-tick";
                tick.title = `${m.label} @ ${m.timestamp.toFixed(2)}s`;
                tick.textContent = m.label;
                tick.style.left = pct + "%";
                tick.addEventListener("click", () => { video.currentTime = m.timestamp; });
                bar.append(tick);
            }
        }
        video.addEventListener("loadedmetadata", layoutMarkers);
        if (video.readyState >= 1) layoutMarkers();
        video.addEventListener("timeupdate", () => {
            timer.textContent = `${fmtTime(video.currentTime)} · frame ${Math.round(video.currentTime * data.fps)}`;
        });

        // Same "capture the real frame fresh at click time" pattern as
        // openPlaybackDialog's own export button.
        box.querySelector("#trake-export-btn").onclick = () => {
            openExportDialog({ kind: "frame", video_id: videoId, frame_idx: Math.round(video.currentTime * data.fps) });
        };
    } catch (e) {
        box.querySelector("#trake-video-wrap").innerHTML = `<div class="status-banner error">${e.message}</div>`;
    }
}
