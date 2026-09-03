// frontend/js/dialogs.js -- modal dialogs. Phase 1: "Nearby frames" (ports
// ui/app.py's show_neighbors, ui/app.py:1334-1360) and single-frame
// playback (ports frame_playback_dialog, ui/app.py:1363-1376), the latter
// now also carrying a one-tick marker bar (same mechanics as TRAKE's,
// marking the keyframe chosen before playback). TRAKE's own multi-event
// marker-bar playback and the Mixed change-weights dialog land in their
// own phases, same file.

import { getNeighbors, getPlayback, getSearchSettings, setSearchSettings } from "./api.js";
import { fmtTime } from "./format.js";
// export-dialog.js now just opens the Export CSV tab (frontend/export.html,
// see its own header) rather than a modal built from openDialog() here --
// no circular import with this file any more.
import { openExportDialog } from "./export-dialog.js";
import {
    getNeighborExtra, MIXED_DEFAULT_LEGS, MIXED_DEFAULT_WEIGHTS,
    MIXED_LEG_DEFS, MIXED_SIGNAL_NAMES, mixedConfig, saveMixedConfig,
} from "./state.js";
import {
    groupByUi, HOVER_ZOOM_MAX, HOVER_ZOOM_MIN, HOVER_ZOOM_STEP,
    QUERY_CHUNK_DEFAULT, QUERY_CHUNK_LABELS, queryChunk, setQueryChunkCache,
    SETTINGS_DEFAULTS, saveSettings, settings, tile, TILE_SIZE_KEYS,
    TILE_SIZES, TOP_K_DEFAULT, TOP_V_DEFAULT,
} from "./settings.js";
import { applyVideoPrefs, bindSpeedShortcut, captureVideoThumbnail } from "./video-controls.js";

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
    // Columns, opening window and expand step all follow the tile-size
    // setting (settings.js's TILE_SIZES): the popup's grid is as wide in
    // tiles as the main result grid, and the counts are picked so it opens
    // three rows full (before + center + after) and expands by two.
    const { neighborsBefore, neighborsAfter, neighborStep: step } = tile();
    const body = document.createElement("div");
    body.innerHTML = `<div class="thumb-caption" style="margin-bottom:0.5rem;">${videoId} — around frame ${centerN}</div>
        <button class="btn" id="nbr-up" style="width:100%;margin-bottom:0.5rem;">▲ ${step} earlier</button>
        <div class="grid nbr-grid" id="nbr-grid"></div>
        <button class="btn" id="nbr-down" style="width:100%;margin-top:0.5rem;">▼ ${step} later</button>`;
    const { box } = openDialog("Nearby frames", body, { wide: true });

    async function refresh() {
        const extra = getNeighborExtra(videoId, centerN);
        const data = await getNeighbors(
            videoId, centerN, neighborsBefore + extra.before, neighborsAfter + extra.after);
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
        getNeighborExtra(videoId, centerN).before += step;
        refresh();
    };
    box.querySelector("#nbr-down").onclick = () => {
        getNeighborExtra(videoId, centerN).after += step;
        refresh();
    };

    await refresh();
}

export async function openPlaybackDialog(videoId, n) {
    const body = document.createElement("div");
    body.innerHTML = `<div class="playback-layout">
        <div class="playback-main">
          <div id="playback-video-wrap">Loading…</div>
          <div id="playback-marker-bar"></div>
        </div>
        <div class="playback-info">
          <div class="thumb-caption">${videoId} — frame ${n}</div>
          <div id="playback-timer" class="playback-timer">--:-- · frame --</div>
          <div id="playback-speed" class="playback-speed" title="&lt; / , slower, &gt; / . faster, 0 resets to 1x">1x</div>
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
        applyVideoPrefs(video);
        bindSpeedShortcut(video, box, box.querySelector("#playback-speed"));

        // Live realtime frame timer -- same fps*currentTime readout TRAKE's
        // marker-bar playback uses (ui/app.py:1440-1442's `timeupdate`
        // handler).
        const timer = box.querySelector("#playback-timer");
        video.addEventListener("timeupdate", () => {
            timer.textContent = `${fmtTime(video.currentTime)} · frame ${Math.round(video.currentTime * data.fps)}`;
        });

        // Single-tick marker bar, same layout/click-to-seek mechanics as
        // TRAKE's #trake-marker-bar, just with exactly one tick: the
        // keyframe (`n`) that was chosen before opening this dialog, at
        // data.start_time.
        const bar = box.querySelector("#playback-marker-bar");
        function layoutMarker() {
            if (!video.duration || !isFinite(video.duration)) return;
            bar.innerHTML = "";
            const pct = Math.max(0, Math.min(100, (data.start_time / video.duration) * 100));
            const tick = document.createElement("div");
            tick.className = "trake-marker-tick";
            tick.title = `frame ${n} @ ${data.start_time.toFixed(2)}s`;
            tick.textContent = `frame ${n}`;
            tick.style.left = pct + "%";
            tick.addEventListener("click", () => { video.currentTime = data.start_time; });
            bar.append(tick);
        }
        video.addEventListener("loadedmetadata", layoutMarker);
        if (video.readyState >= 1) layoutMarker();

        // Exports whatever real frame the video is currently at (paused or
        // not), computed fresh at click time -- not read off the timer's
        // text, which is just a display of the same arithmetic. No
        // keyframe n involved at all, so this always opens as a TRAKE
        // export (see export-ui.js's {kind:"frame"} handling). Carries the
        // current playback position (seconds) and a thumbnail snapshot
        // across the tab handoff too: the Export tab's curation video
        // seeks to the same spot instead of restarting at 0:00, and the
        // seeded TRAKE event gets a real preview instead of "no preview"
        // (see export-ui.js's addTrakeEventFromTrigger/loadCurationVideo).
        box.querySelector("#playback-export-btn").onclick = () => {
            openExportDialog({
                kind: "frame", video_id: videoId,
                frame_idx: Math.round(video.currentTime * data.fps),
                current_time: video.currentTime,
                thumbnail: captureVideoThumbnail(video),
            });
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

// Display settings dialog -- the ⚙ button in the sidebar's signal rows.
// Same staged-copy/Defaults/Cancel/Save shape as openWeightsDialog above:
// nothing is committed (or persisted) until Save, so Cancel really does
// discard. `onSave` re-runs the current search -- every control here can
// change what a search returns or how it's grouped.
//
// Three kinds of control share the form: the saved settings themselves
// (settings.js); mirrors of the sidebar's Top-K/Top-V/Top-G boxes, which stay
// on the sidebar and stay session state -- the dialog reads them on open and
// writes back only the ones actually changed, so it never clobbers a
// hand-typed value it didn't touch; and one backend setting, query chunking,
// which is fetched on open and POSTed on Save.
export function openSettingsDialog(onSave) {
    const staged = { ...settings };
    // Drawn from the cached value immediately, then corrected by the fetch
    // below -- the dialog must never present a chunking mode the backend
    // isn't actually using. `loadedChunk` is what the backend said, so Save
    // can tell a real change from a no-op and skip the POST.
    let stagedChunk = queryChunk.strategy;
    let loadedChunk = queryChunk.strategy;
    let chunkTouched = false;

    const TOP_BOXES = [
        { id: "top-k", label: "Top-K", title: "Candidates fetched per search." },
        { id: "top-v", label: "Top-V", title: "Videos kept (TRAKE)." },
        { id: "top-g", label: "Top-G", title: "Frames kept per video after per-video drill-down (Hierarchy)." },
    ];
    const sidebarInput = (id) => document.getElementById(id);
    // All three are always offered here, even though the sidebar shows Top-V
    // only on TRAKE and Top-G only on Hierarchy: the inputs (and their
    // values) exist either way, so this is the one place to set them up
    // before switching to the signal that uses them.
    const initialTops = Object.fromEntries(TOP_BOXES.map((b) => [b.id, sidebarInput(b.id).value]));
    const stagedTops = { ...initialTops };
    // A hand-typed Top-G outranks the tile size's default below.
    let topGTouched = false;

    const body = document.createElement("div");
    body.innerHTML = `<div class="settings-row">
          <label for="set-zoom" title="How far a result thumbnail scales up while hovered.">Hover zoom</label>
          <input type="range" id="set-zoom" min="${HOVER_ZOOM_MIN}" max="${HOVER_ZOOM_MAX}" step="${HOVER_ZOOM_STEP}">
          <span class="settings-value" id="set-zoom-value"></span>
        </div>
        <div class="settings-row">
          <label title="Thumbnail size everywhere: fewer, bigger tiles per row (and matching &quot;show more&quot; steps) at Large.">Tiles display size</label>
          <div class="segmented" id="set-tile">
            ${TILE_SIZE_KEYS.map((key) => `<button type="button" data-tile="${key}">${TILE_SIZES[key].label}</button>`).join("")}
          </div>
        </div>
        <div class="settings-row">
          <label title="The same boxes as the sidebar's -- changed here, they change there.">Result counts</label>
          <div class="settings-tops">
            ${TOP_BOXES.map((b) => `<div class="settings-top-box">
              <label for="set-${b.id}" title="${b.title}">${b.label}</label>
              <input type="number" id="set-${b.id}" min="1" step="1">
            </div>`).join("")}
          </div>
        </div>
        <div class="settings-row">
          <label title="SigLIP2's text tower reads at most 64 tokens. A longer query has to be split -- this is what happens to the pieces. Applies to the backend behind this tab, not just this browser.">Long-query chunking</label>
          <div class="segmented" id="set-chunk">
            ${Object.entries(QUERY_CHUNK_LABELS).map(([key, v]) =>
              `<button type="button" data-chunk="${key}" title="${v.title}">${v.label}</button>`).join("")}
          </div>
        </div>
        <div class="settings-row">
          <label>Result display</label>
          <div class="settings-checks">
            <label class="settings-check" id="set-group-row">
              <input type="checkbox" id="set-group"> <span id="set-group-label"></span>
            </label>
            <label class="settings-check">
              <input type="checkbox" id="set-fulltext"> Show full text
            </label>
          </div>
        </div>
        <hr class="divider">
        <div class="settings-actions">
          <button class="btn" id="set-defaults">Set to defaults</button>
          <button class="btn" id="set-cancel">Cancel</button>
          <button class="btn btn-primary" id="set-save">Save</button>
        </div>`;
    const { overlay, box } = openDialog("Settings", body);

    const zoom = box.querySelector("#set-zoom");
    const zoomValue = box.querySelector("#set-zoom-value");
    const groupCheck = box.querySelector("#set-group");
    const fullTextCheck = box.querySelector("#set-fulltext");

    // Hierarchy/TRAKE don't offer a group-by toggle at all, and Summary
    // relabels it -- one shared toggle, presented per the mounted signal
    // (settings.js's groupByUi).
    box.querySelector("#set-group-row").style.display = groupByUi.visible ? "flex" : "none";
    box.querySelector("#set-group-label").textContent = groupByUi.label;

    function renderStaged() {
        zoom.value = staged.hoverZoom;
        zoomValue.textContent = `${Number(staged.hoverZoom).toFixed(1)}×`;
        // Exactly one tile size active at a time -- clicking one clears its
        // siblings (unlike the sidebar's scope segmented control, this one
        // can't drop to zero selected).
        box.querySelectorAll("#set-tile button").forEach((btn) => {
            btn.classList.toggle("active", btn.dataset.tile === staged.tileSize);
        });
        box.querySelectorAll("#set-chunk button").forEach((btn) => {
            btn.classList.toggle("active", btn.dataset.chunk === stagedChunk);
        });
        groupCheck.checked = staged.groupByVideo;
        fullTextCheck.checked = staged.showFullText;
        for (const b of TOP_BOXES) box.querySelector(`#set-${b.id}`).value = stagedTops[b.id];
    }
    renderStaged();

    zoom.oninput = () => {
        staged.hoverZoom = Math.round(parseFloat(zoom.value) * 10) / 10;
        zoomValue.textContent = `${staged.hoverZoom.toFixed(1)}×`;
    };
    // The live backend value, in case another tab (or a restart) moved it
    // since this page loaded. Silent on failure: an unreachable /api/settings
    // leaves the cached value showing rather than blocking the whole dialog.
    getSearchSettings().then(({ query_chunk_strategy }) => {
        loadedChunk = setQueryChunkCache(query_chunk_strategy);
        // Don't stomp a choice already clicked while the fetch was in flight.
        if (!chunkTouched) stagedChunk = loadedChunk;
        renderStaged();
    }).catch(() => { /* keep showing the cached value */ });

    box.querySelectorAll("#set-chunk button").forEach((btn) => {
        btn.onclick = () => { stagedChunk = btn.dataset.chunk; chunkTouched = true; renderStaged(); };
    });
    box.querySelectorAll("#set-tile button").forEach((btn) => {
        btn.onclick = () => {
            staged.tileSize = btn.dataset.tile;
            // Top-G's default is a property of the tile size, so picking a
            // size moves the box with it -- unless the user typed their own.
            if (!topGTouched) stagedTops["top-g"] = String(TILE_SIZES[staged.tileSize].topG);
            renderStaged();
        };
    });
    for (const b of TOP_BOXES) {
        box.querySelector(`#set-${b.id}`).oninput = (e) => {
            stagedTops[b.id] = e.target.value;
            if (b.id === "top-g") topGTouched = true;
        };
    }
    groupCheck.onchange = () => { staged.groupByVideo = groupCheck.checked; };
    fullTextCheck.onchange = () => { staged.showFullText = fullTextCheck.checked; };

    box.querySelector("#set-defaults").onclick = () => {
        Object.assign(staged, SETTINGS_DEFAULTS);
        stagedChunk = QUERY_CHUNK_DEFAULT;
        chunkTouched = true;
        const topDefaults = {
            "top-k": TOP_K_DEFAULT,
            "top-v": TOP_V_DEFAULT,
            "top-g": TILE_SIZES[staged.tileSize].topG,
        };
        for (const b of TOP_BOXES) stagedTops[b.id] = String(topDefaults[b.id]);
        topGTouched = false;
        renderStaged();
    };
    box.querySelector("#set-cancel").onclick = () => overlay.remove();
    box.querySelector("#set-save").onclick = () => {
        saveSettings(staged);
        // Only the boxes actually changed are written back -- including
        // Top-G when a new tile size moved it (see the size buttons above),
        // whether or not the current signal shows Top-G in the sidebar.
        for (const b of TOP_BOXES) {
            if (stagedTops[b.id] !== initialTops[b.id]) sidebarInput(b.id).value = stagedTops[b.id];
        }
        overlay.remove();
        // The chunking mode is the one setting that has to reach the backend
        // before the re-run, or the search would still use the old one. On a
        // failed POST the cache is left alone and the search runs unchanged,
        // rather than the UI claiming a mode the backend never took.
        if (stagedChunk !== loadedChunk) {
            setSearchSettings({ query_chunk_strategy: stagedChunk })
                .then(({ query_chunk_strategy }) => setQueryChunkCache(query_chunk_strategy))
                .catch(() => { /* backend kept its old mode; so do we */ })
                .then(() => { if (onSave) onSave(); });
            return;
        }
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
          <div id="trake-speed" class="playback-speed" title="&lt; / , slower, &gt; / . faster, 0 resets to 1x">1x</div>
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
        applyVideoPrefs(video);
        bindSpeedShortcut(video, box, box.querySelector("#trake-speed"));

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
        // openPlaybackDialog's own export button, including the current-
        // time/thumbnail handoff (see its comment above).
        box.querySelector("#trake-export-btn").onclick = () => {
            openExportDialog({
                kind: "frame", video_id: videoId,
                frame_idx: Math.round(video.currentTime * data.fps),
                current_time: video.currentTime,
                thumbnail: captureVideoThumbnail(video),
            });
        };
    } catch (e) {
        box.querySelector("#trake-video-wrap").innerHTML = `<div class="status-banner error">${e.message}</div>`;
    }
}
