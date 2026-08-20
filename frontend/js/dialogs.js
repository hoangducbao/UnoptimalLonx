// frontend/js/dialogs.js -- modal dialogs. Phase 1: "Nearby frames" (ports
// ui/app.py's show_neighbors, ui/app.py:1334-1360) and single-frame
// playback (ports frame_playback_dialog, ui/app.py:1363-1376). TRAKE's
// marker-bar playback and the Mixed change-weights dialog land in their
// own phases, same file.

import { getNeighbors, getPlayback } from "./api.js";
import { getNeighborExtra } from "./state.js";

const root = document.getElementById("dialog-root");

function openDialog(title, bodyEl, { wide = false } = {}) {
    const overlay = document.createElement("div");
    overlay.className = "dialog-overlay";
    const box = document.createElement("div");
    box.className = "dialog-box" + (wide ? " wide" : "");
    const closeBtn = document.createElement("button");
    closeBtn.className = "dialog-close";
    closeBtn.textContent = "✕";
    closeBtn.onclick = () => overlay.remove();
    overlay.onclick = (e) => { if (e.target === overlay) overlay.remove(); };

    const h3 = document.createElement("h3");
    h3.textContent = title;

    box.append(closeBtn, h3, bodyEl);
    overlay.append(box);
    root.append(overlay);
    return { overlay, box };
}

export async function openNeighborsDialog(videoId, centerN) {
    const body = document.createElement("div");
    body.innerHTML = `<div class="thumb-caption" style="margin-bottom:0.5rem;">${videoId} — around frame ${centerN}</div>
        <button class="btn" id="nbr-up" style="width:100%;margin-bottom:0.5rem;">▲ 10 earlier</button>
        <div class="grid" id="nbr-grid" style="grid-template-columns:repeat(5,1fr);"></div>
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
            cell.innerHTML = f.exists
                ? `<div class="thumb-wrap"><img src="${f.thumbnail_url}"></div>`
                : `<div class="thumb-missing">(missing)</div>`;
            const cap = document.createElement("div");
            cap.className = "thumb-caption";
            cap.innerHTML = f.is_center ? `<b>${f.n}</b>` : String(f.n);
            cell.append(cap);
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
    body.innerHTML = `<div class="thumb-caption" style="margin-bottom:0.5rem;">${videoId} — frame ${n}</div>
        <div id="playback-video-wrap">Loading…</div>
        <div id="playback-timer" style="font-family:monospace;font-size:0.9rem;margin-top:6px;">--:-- · frame --</div>`;
    const { box } = openDialog("Video playback", body);

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
    } catch (e) {
        box.querySelector("#playback-video-wrap").innerHTML =
            `<div class="status-banner error">${e.message}</div>`;
    }
}
