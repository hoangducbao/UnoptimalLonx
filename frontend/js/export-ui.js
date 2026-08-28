// frontend/js/export-ui.js -- the actual Export CSV UI: query-type
// segmented control, confirmed/unconfirmed answer curation, Neighbours/
// Similars preview grids (KIS/VQA only), TRAKE's per-video event curation.
// Mounted by export-page.js into the standalone Export CSV tab (frontend/export.html)
// -- this module knows nothing about being in a separate tab: the host
// supplies a plain container element, a `getCandidates` accessor (so
// "Similars" can read a *different* tab's live search results without this
// module importing that tab's state.js), and an `onDone()` callback for
// "the user cancelled" (a completed export leaves the form in place instead
// -- see #exp-export's handler -- so onDone is never called for that case).
//
// Two bodies depending on query type:
//   KIS/VQA -- one frame's worth of query-answer curation (confirmed/
//     unconfirmed) plus a two-section preview (nearest-by-time
//     "Neighbours", already-ranked "Similars"). POSTs to /api/export and
//     triggers the CSV download directly.
//   TRAKE -- no confirmed/unconfirmed distinction at all any more. A
//     curate -> cache -> merge flow instead:
//       1. Curate one video at a time: an inline <video> preview (loaded
//          by typing a video id, or seeded from `trigger`) plus an "Add"
//          button that captures whatever frame is currently playing into
//          an ordered event list (drag to reorder, ✕ to remove). No
//          Neighbours/Similars preview for TRAKE -- the Frame ID box is
//          the only other way to add an event, besides the video itself.
//       2. "Generate rows" POSTs that video's {video_id, frame_idxs} to
//          /api/export/trake-rows and caches the <=100 returned candidate
//          sequences client-side, keyed by video_id (`s.trake.cache`
//          below) -- repeatable for as many candidate videos as the human
//          wants to compare, each just adding another entry.
//       3. Export: the human checks which cached videos to include and
//          their priority order; the rows are interleaved client-side (no
//          backend round-trip, no re-reading anything) into one <=100-row
//          set and POSTed to /api/export/trake-write, which only formats
//          + returns CSV text for already-resolved rows -- the one file
//          this whole flow ever writes to disk.
//     All of one video's events still share that one video_id (the AIC
//     TRAKE row format is one video per row) -- switching the curation
//     panel to a different video starts a fresh event list, independent
//     per-video cache entries are what let several candidate videos
//     coexist for the merge step.
//
// `trigger` shapes:
//   {kind: "flat", video_id, n}         -- any non-TRAKE signal's result card
//   {kind: "trake", candidate}          -- a TRAKE candidate card
//   {kind: "frame", video_id, frame_idx} -- a raw native frame from a video
//     playback dialog, no keyframe n at all (TRAKE-only: KIS/VQA need an n
//     for the backend's flat CSV path, which this trigger doesn't have).

import { exportCsv, getExportFrame, getExportNeighbors, getPlayback, getTrakeRows, writeTrakeCsv } from "./api.js";

const NEIGHBOUR_COUNT_EXPORT = 10; // fixed row-generation window, independent of preview expand state
const PREVIEW_PAGE = 12; // 3x4 grid per preview section (export-preview-grid is 4 columns wide)

function freshState(trigger) {
    const isFlat = trigger.kind === "flat";
    const seed = isFlat ? { video_id: trigger.video_id, n: trigger.n } : null;

    return {
        trigger,
        queryType: isFlat ? "KIS" : "TRAKE", // "trake"/"frame" triggers default to TRAKE -- "frame" has no n for KIS/VQA's flat CSV path
        name: "",
        confirmed: true,
        answerText: "",
        answerFrame: seed,          // confirmed-mode single answer (KIS/VQA)
        answers: seed ? [seed] : [], // unconfirmed-mode ordered list, pre-seeded per spec ("at least 1 frame")
        frameInfo: new Map(),        // "vid|n" -> {frame_idx, thumbnail_url} | "pending"
        neighbourFrames: null,       // cached /api/export/neighbors result (grows with `neighboursShown`)
        neighboursShown: PREVIEW_PAGE,
        similarsShown: PREVIEW_PAGE,
        dragIndex: null,
        // TRAKE: no confirmed/unconfirmed distinction any more -- one
        // per-video curation session (video + ordered event list, each a
        // native frame_idx with its own add-time thumbnail) feeds a
        // "Generate rows" call whose <=100 candidate sequences are cached
        // here per video_id; a final merge step interleaves however many
        // cached videos the human picked, in priority order, into one CSV.
        // See freshTrakeState() below and the "TRAKE: curate one video's
        // events..." section further down for the rest.
        trake: freshTrakeState(),
    };
}

function freshTrakeState() {
    return {
        videoId: null,
        events: [],       // [{frame_idx, thumbnail}], in sequence order (event 1..N)
        dragIndex: null,
        videoEl: null,     // the curation panel's live <video>, for Add/capture
        fps: 25,
        // video_id -> {frameIdxs: [...], rows: [[f1..fN], ...]} -- one
        // entry per "Generate rows" click; overwritten if regenerated for
        // the same video_id.
        cache: new Map(),
        mergeOrder: [],    // video_ids, in merge priority order (checked ones only need be present)
        mergeChecked: new Set(),
        mergeDragIndex: null,
    };
}

function frameKey(f) { return `${f.video_id}|${f.n}`; }

// The Video ID/Frame ID typing boxes: `L21_V001`-shaped video id,
// plain-integer frame number (the leading zeros in the "001" placeholder
// are just display convention -- parseInt handles them fine either way).
// The same parser serves both meanings the Frame ID box can have: a
// keyframe n (KIS/VQA, resolved server-side via /api/export/frame) or a
// raw native frame_idx (TRAKE, used directly, no resolution needed) --
// both are just positive integers as typed. Return null on a malformed
// box so the caller can show one "correct format" error rather than
// letting a bad value reach the backend as a confusing 404/422.
function parseVideoIdInput(str) {
    const s = (str || "").trim();
    return /^L\d+_V\d+$/i.test(s) ? s.toUpperCase() : null;
}
function parseFrameIdInput(str) {
    const s = (str || "").trim();
    if (!/^\d+$/.test(s)) return null;
    const n = parseInt(s, 10);
    return n > 0 ? n : null;
}

// AIC submission naming: query-p2-<#>-<type>.csv, matching the real
// submission/*.csv samples in the repo -- note VQA's type slug is "qa",
// not "vqa".
const TYPE_SLUG = { KIS: "kis", VQA: "qa", TRAKE: "trake" };
function queryFilename(queryType, name) {
    return `query-p2-${name}-${TYPE_SLUG[queryType]}`;
}

// Kicks off a fetch for `f`'s frame_idx if it isn't cached yet; callers
// that already have a cached value use it directly when building their
// HTML (see renderAnswerContent()) rather than going through here, so
// `onReady` only ever fires asynchronously, after a genuine network round
// trip -- never synchronously/re-entrantly. (It used to fire synchronously
// for already-cached frames too, which re-entered renderAnswerContent()
// from inside its own answers.forEach(), recursing once per already-cached
// card; with more than a couple of cards that overflowed the call stack,
// and since the throw happened inside this function's own promise chain,
// its .catch() silently swallowed it and deleted the just-fetched cache
// entry -- so a newly-added card's frame_idx never appeared, stuck on "…"
// forever, with no visible error.)
function ensureFrameInfo(s, f, onReady) {
    const key = frameKey(f);
    const cached = s.frameInfo.get(key);
    if (cached) return; // already resolved-and-rendered, or a fetch is already in flight
    s.frameInfo.set(key, "pending");
    getExportFrame(f.video_id, f.n).then((info) => {
        s.frameInfo.set(key, info);
        onReady(info);
    }).catch(() => { s.frameInfo.delete(key); });
}

export function buildExportUI(container, trigger, { getCandidates, onDone }) {
    const s = freshState(trigger);

    const box = document.createElement("div");
    box.className = "export-dialog";
    box.innerHTML = `
        <h2>Export CSV</h2>
        <div class="status-banner" id="exp-status" style="display:none;"></div>
        <div class="export-topbar" id="exp-topbar">
          <div class="segmented" id="exp-segmented">
            <button type="button" data-type="KIS">KIS</button>
            <button type="button" data-type="VQA">VQA</button>
            <button type="button" data-type="TRAKE">TRAKE</button>
          </div>
          <input type="number" id="exp-name" min="1" step="1" placeholder="Query no.">
          <div class="export-change-fields" id="exp-change-fields">
            <input type="text" id="exp-change-video" placeholder="Vid ID">
            <input type="text" id="exp-change-frame" placeholder="Frame ID">
            <button class="btn" id="exp-change-btn" type="button">Change</button>
          </div>
          <div class="export-actions">
            <button class="btn" id="exp-cancel">Cancel</button>
            <button class="btn btn-primary" id="exp-export">⬇ Export</button>
          </div>
        </div>
        <div class="checkbox-row" id="exp-confirmed-row">
          <input type="checkbox" id="exp-confirmed" checked>
          <label for="exp-confirmed" style="margin:0;">Confirmed</label>
        </div>
        <div id="exp-flat-body">
          <div class="export-answer-area">
            <div id="exp-answer-content"></div>
            <input type="text" id="exp-answer-text" placeholder="VQA answer" style="display:none;">
          </div>
        </div>
        <div id="exp-trake-body" style="display:none;">
          <div id="exp-trake-content"></div>
        </div>
        <div class="export-preview-area" id="exp-preview-area">
          <div class="export-preview-section">
            <div class="export-preview-header"><b>Neighbours</b> <span class="thumb-caption muted">(nearest keyframes by time)</span></div>
            <div class="grid export-preview-grid" id="exp-nbr-grid"></div>
            <button class="btn" id="exp-nbr-more">Show 12 more</button>
          </div>
          <div class="export-preview-section">
            <div class="export-preview-header"><b>Similars</b> <span class="thumb-caption muted">(this query's ranked results)</span></div>
            <div class="grid export-preview-grid" id="exp-sim-grid"></div>
            <button class="btn" id="exp-sim-more">Show 12 more</button>
          </div>
        </div>`;
    container.innerHTML = "";
    container.append(box);

    const el = (sel) => box.querySelector(sel);

    // Status banner doubles as the validation-error banner (used throughout
    // below) and the post-export confirmation (item: "leave it there with a
    // message, user can reexport if they want" -- see #exp-export's handler,
    // which shows success here rather than tearing the form down).
    function showStatus(message, kind = "error") {
        const banner = el("#exp-status");
        banner.className = `status-banner ${kind}`;
        banner.textContent = message;
        banner.style.display = "block";
    }
    function clearStatus() {
        el("#exp-status").style.display = "none";
    }

    function renderTypeVisibility() {
        el("#exp-segmented").querySelectorAll("button").forEach((btn) => {
            btn.classList.toggle("active", btn.dataset.type === s.queryType);
        });
        const isTrake = s.queryType === "TRAKE";
        el("#exp-trake-body").style.display = isTrake ? "block" : "none";
        el("#exp-flat-body").style.display = isTrake ? "none" : "block";
        // No confirmed/unconfirmed distinction for TRAKE any more (see
        // module docstring) -- the checkbox only means something for
        // KIS/VQA.
        el("#exp-confirmed-row").style.display = isTrake ? "none" : "flex";
        // No Neighbours/Similars preview for TRAKE at all -- per spec,
        // TRAKE events have no "similar" pool to search (only a picked
        // frame's own keyframe/native-distance neighbours, which is what
        // row generation already computes server-side); events come from
        // the curation panel's video playback, the Frame ID box, or the
        // seeding trigger only. renderPreview() itself skips fetching for
        // TRAKE too, not just this visibility toggle.
        el("#exp-preview-area").style.display = isTrake ? "none" : "flex";

        // Frame ID/Change is repurposed for TRAKE, not hidden: a native
        // frame number needs no keyframe lookup, so it means "add an
        // event to the video currently being curated" instead of
        // "replace/add the answer frame". Video ID is locked to that
        // video (switching video is its own explicit action inside the
        // TRAKE curation panel below) so the user can't typo an event
        // into the wrong video's sequence.
        el("#exp-change-frame").placeholder = isTrake ? "Frame ID (real frame)" : "Frame ID";
        el("#exp-change-btn").textContent = isTrake ? "Add event" : "Change";
        const changeVideo = el("#exp-change-video");
        changeVideo.readOnly = isTrake;
        changeVideo.value = isTrake ? (s.trake.videoId || "") : "";
        changeVideo.title = isTrake ? "Switch curation video from the TRAKE panel below" : "";

        // Same typing box either way -- unconfirmed mode used to disable
        // this with an "LLM needed" placeholder (answering unconfirmed
        // VQA queries was meant to be automated later), but that's no
        // longer the plan: a human types the answer regardless of mode.
        const isVqa = s.queryType === "VQA";
        const answerText = el("#exp-answer-text");
        answerText.style.display = isVqa ? "block" : "none";
        answerText.disabled = false;
        answerText.placeholder = "VQA answer";
    }

    function frameCardHtml(f, info, { removable = false, index = null } = {}) {
        const thumb = info && info !== "pending"
            ? `<img src="${info.thumbnail_url}" loading="lazy">`
            : `<div class="thumb-missing">…</div>`;
        const frameIdx = info && info !== "pending" ? info.frame_idx : "…";
        const removeBtn = removable ? `<button class="icon-btn export-remove-btn" title="Remove" data-index="${index}">✕</button>` : "";
        return `<div class="export-answer-card" ${removable ? `draggable="true" data-index="${index}"` : ""}>
            <div class="thumb-wrap thumb-wrap-static">${thumb}</div>
            <div class="thumb-caption"><b>${f.video_id}</b> · keyframe ${f.n}</div>
            <div class="thumb-caption muted">real frame ${frameIdx}</div>
            ${removeBtn}
        </div>`;
    }

    function renderAnswerContent() {
        const content = el("#exp-answer-content");
        if (s.confirmed) {
            if (!s.answerFrame) {
                content.innerHTML = `<div class="status-banner info">No frame selected -- open this from a result card's ★ button.</div>`;
                return;
            }
            const info = s.frameInfo.get(frameKey(s.answerFrame));
            content.innerHTML = `<div class="export-answer-list">${frameCardHtml(s.answerFrame, info)}</div>`;
            ensureFrameInfo(s, s.answerFrame, () => renderAnswerContent());
        } else {
            if (!s.answers.length) {
                content.innerHTML = `<div class="status-banner info">Add at least one frame from the preview below.</div>`;
            } else {
                content.innerHTML = `<div class="export-answer-list">${s.answers.map((f, i) => {
                    const info = s.frameInfo.get(frameKey(f));
                    return frameCardHtml(f, info, { removable: true, index: i });
                }).join("")}</div>`;
                s.answers.forEach((f) => ensureFrameInfo(s, f, () => renderAnswerContent()));
            }
            wireAnswerDnd();
        }
    }

    function wireAnswerDnd() {
        for (const card of el("#exp-answer-content").querySelectorAll(".export-answer-card")) {
            const i = Number(card.dataset.index);
            card.addEventListener("dragstart", () => { s.dragIndex = i; });
            card.addEventListener("dragover", (e) => e.preventDefault());
            card.addEventListener("drop", (e) => {
                e.preventDefault();
                if (s.dragIndex === null || s.dragIndex === i) return;
                const [moved] = s.answers.splice(s.dragIndex, 1);
                s.answers.splice(i, 0, moved);
                s.dragIndex = null;
                renderAnswerContent();
            });
            const removeBtn = card.querySelector(".export-remove-btn");
            if (removeBtn) removeBtn.onclick = () => {
                s.answers.splice(i, 1);
                renderAnswerContent();
            };
        }
    }

    function isInAnswers(f) {
        return s.answers.some((a) => a.video_id === f.video_id && a.n === f.n);
    }

    function addToAnswers(f) {
        if (isInAnswers(f)) return;
        s.answers.push({ video_id: f.video_id, n: f.n });
        // Mirror into the confirmed single-frame slot only while there's
        // still exactly one frame in the unconfirmed list -- once a second
        // is added, "the" answer frame is ambiguous, so stop syncing
        // rather than guess which one confirmed mode should show.
        if (s.answers.length === 1) s.answerFrame = s.answers[0];
        renderAnswerContent();
        renderPreview();
    }

    function previewCardHtml(f, { addable, replaceable }) {
        const already = addable && isInAnswers(f);
        const isCurrent = replaceable && s.answerFrame && s.answerFrame.video_id === f.video_id && s.answerFrame.n === f.n;
        const addBtn = addable
            ? `<button class="icon-btn export-add-btn${already ? " added" : ""}" title="${already ? "Already added" : "Add to answer(s)"}" data-video-id="${f.video_id}" data-n="${f.n}">${already ? "✓" : "+"}</button>`
            : "";
        // Confirmed mode: picking a preview frame replaces the single
        // answer frame instead of adding to a list. Reuses .export-add-btn's
        // CSS (same corner position, same .added accent) and is told apart
        // from the add button by data-replace, not class.
        const replaceBtn = replaceable
            ? `<button class="icon-btn export-add-btn${isCurrent ? " added" : ""}" title="${isCurrent ? "Current answer frame" : "Use as answer frame"}" data-video-id="${f.video_id}" data-n="${f.n}" data-replace="1"${isCurrent ? " disabled" : ""}>${isCurrent ? "✓" : "⇄"}</button>`
            : "";
        return `<div class="thumb-cell">
            <div class="thumb-wrap thumb-wrap-static"><img src="${f.thumbnail_url}" loading="lazy"></div>
            <div class="thumb-caption"><b>${f.video_id}</b> · frame ${f.n}</div>
            ${addBtn}${replaceBtn}
        </div>`;
    }

    // KIS/VQA only -- TRAKE has no Neighbours/Similars preview at all (see
    // renderTypeVisibility's #exp-preview-area toggle and module
    // docstring: a TRAKE pick's only "similar" pool is what row generation
    // already computes server-side, not something to browse here).
    async function renderPreview() {
        if (s.queryType === "TRAKE") return;

        const addable = !s.confirmed;
        const replaceable = s.confirmed;

        // Neighbours -- nearest keyframes by time to the trigger frame.
        const nbrGrid = el("#exp-nbr-grid");
        if (s.trigger.kind !== "flat") {
            nbrGrid.innerHTML = `<div class="status-banner info">No source frame to find neighbours of.</div>`;
        } else {
            if (!s.neighbourFrames || s.neighbourFrames.length < s.neighboursShown) {
                nbrGrid.innerHTML = `<div class="status-banner info">Loading…</div>`;
                try {
                    const data = await getExportNeighbors(s.trigger.video_id, s.trigger.n, s.neighboursShown);
                    s.neighbourFrames = data.frames;
                } catch (e) {
                    nbrGrid.innerHTML = `<div class="status-banner error">${e.message}</div>`;
                    return;
                }
            }
            const frames = s.neighbourFrames.slice(0, s.neighboursShown).map((f) => ({ ...f, video_id: s.trigger.video_id }));
            nbrGrid.innerHTML = frames.map((f) => previewCardHtml(f, { addable, replaceable })).join("") || `<div class="status-banner info">No neighbours found.</div>`;
            el("#exp-nbr-more").style.display = s.neighbourFrames.length >= s.neighboursShown ? "block" : "none";
        }

        // Similars -- the query's own already-fetched, already-ranked results.
        // These may be TRAKE-shaped ({video_id, events}, no .n) if the
        // opener tab's last search was a real TRAKE search -- previewCardHtml
        // needs a flat {video_id, n} shape, so fall back to a plain message
        // rather than rendering broken cards.
        const candidates = getCandidates();
        const simGrid = el("#exp-sim-grid");
        if (candidates.length && !("n" in candidates[0])) {
            simGrid.innerHTML = `<div class="status-banner info">Last search wasn't a flat-result signal -- no Similars to preview.</div>`;
            el("#exp-sim-more").style.display = "none";
        } else {
            const similars = candidates.slice(0, s.similarsShown);
            simGrid.innerHTML = similars.map((c) => previewCardHtml(c, { addable, replaceable }))
                .join("") || `<div class="status-banner info">No results from the last search.</div>`;
            el("#exp-sim-more").style.display = candidates.length > s.similarsShown ? "block" : "none";
        }

        if (addable) {
            box.querySelectorAll(".export-add-btn:not([data-replace])").forEach((btn) => {
                btn.onclick = () => addToAnswers({ video_id: btn.dataset.videoId, n: Number(btn.dataset.n) });
            });
        }
        if (replaceable) {
            box.querySelectorAll(".export-add-btn[data-replace]").forEach((btn) => {
                btn.onclick = () => applyChangedFrame({ video_id: btn.dataset.videoId, n: Number(btn.dataset.n) });
            });
        }
    }

    // Item 4 (pick a preview frame to replace) and item 5 (type a video/
    // frame id and hit Change) both funnel through here: confirmed mode
    // replaces the single answer frame, unconfirmed mode adds to the answer
    // list (same as the preview's own "+" button).
    //
    // Confirmed's answerFrame and unconfirmed's answers list also get
    // synced here -- a confirmed-mode edit collapses the answers list down
    // to that one frame, since at that point there's exactly one frame in
    // play and both views should agree on it (otherwise toggling Confirmed
    // off would silently revert to whatever the tab was originally seeded
    // with instead of the just-changed frame).
    function applyChangedFrame(f) {
        if (s.confirmed) {
            if (s.answerFrame && frameKey(s.answerFrame) === frameKey(f)) return;
            s.answerFrame = f;
            s.answers = [f];
            renderAnswerContent();
            renderPreview();
        } else {
            addToAnswers(f);
        }
    }

    // --- TRAKE: curate one video's events -> cache its generated rows ->
    // merge however many cached videos into the final export -----------

    function fmtTime(t) {
        const mm = String(Math.floor(t / 60)).padStart(2, "0");
        const ss = (t % 60).toFixed(2).padStart(5, "0");
        return `${mm}:${ss}`;
    }

    // Grabs a JPEG data URL of whatever frame `video` is showing right
    // now -- this is the "cache the thumbnail at add-time" half of the
    // spec, since a raw native frame has no existing thumbnail file to
    // point at the way a keyframe does. The video element is same-origin
    // (served from this app's own /media/video mount), so the canvas
    // isn't tainted; still guarded in case a frame isn't decoded yet.
    function captureVideoThumbnail(video) {
        try {
            const w = video.videoWidth || 320, h = video.videoHeight || 180;
            const canvas = document.createElement("canvas");
            canvas.width = 160;
            canvas.height = Math.round(160 * (h / w));
            canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
            return canvas.toDataURL("image/jpeg", 0.7);
        } catch (e) {
            return null;
        }
    }

    // Loads (or switches the curation panel to) a video: fetches playback
    // info and builds a fresh <video>. Switching to a *different* video
    // than the one currently being curated starts a clean event list --
    // any cache entry already generated for either video is untouched
    // (cache entries persist independently of what's in the live curation
    // panel, see generateRowsForCurationVideo()).
    async function loadCurationVideo(videoId) {
        videoId = (videoId || "").trim().toUpperCase();
        if (!videoId) return;
        if (videoId !== s.trake.videoId) {
            s.trake.videoId = videoId;
            s.trake.events = [];
            renderEventList();
        }
        clearStatus();
        renderTypeVisibility(); // keeps the topbar's locked Video ID display in sync
        const wrap = el("#trake-video-wrap");
        if (wrap) wrap.innerHTML = `<div class="status-banner info">Loading…</div>`;
        try {
            const data = await getPlayback(videoId);
            if (el("#trake-video-wrap") !== wrap) return; // panel torn down mid-fetch (query type switched away)
            s.trake.fps = data.fps;
            wrap.innerHTML = "";
            const video = document.createElement("video");
            video.src = data.video_url;
            video.controls = true;
            wrap.append(video);
            s.trake.videoEl = video;
            const timer = el("#trake-cur-timer");
            video.addEventListener("timeupdate", () => {
                timer.textContent = `${fmtTime(video.currentTime)} · frame ${Math.round(video.currentTime * s.trake.fps)}`;
            });
        } catch (e) {
            s.trake.videoEl = null;
            if (wrap) wrap.innerHTML = `<div class="status-banner error">${e.message}</div>`;
        }
    }

    // Adds one event to the video currently being curated -- from the
    // inline "Add current frame" button (f.thumbnail already captured) or
    // the repurposed Frame ID/Change row (raw frame_idx, no thumbnail).
    // Enforces the one hard constraint: a TRAKE export row is exactly one
    // video, so a frame from a different video is rejected rather than
    // silently starting a second, unrepresentable sequence.
    function addTrakeEvent(f) {
        if (s.trake.videoId && f.video_id !== s.trake.videoId) {
            showStatus(`Currently curating ${s.trake.videoId} -- this frame is from a different video. Switch videos above first if you meant to add it there.`);
            return false;
        }
        if (!s.trake.videoId) s.trake.videoId = f.video_id;
        s.trake.events.push({ frame_idx: f.frame_idx, thumbnail: f.thumbnail ?? null });
        clearStatus();
        renderEventList();
        return true;
    }

    // Resolves a keyframe n (not a raw frame_idx) to its real frame_idx +
    // thumbnail before adding -- used to seed the curation panel from a
    // "flat" trigger (any non-TRAKE signal's ★, which only carries n, no
    // frame_idx) via the same lookup the rest of the app already does.
    function addTrakeEventFromN(videoId, n) {
        if (s.trake.videoId && videoId !== s.trake.videoId) {
            showStatus(`Currently curating ${s.trake.videoId} -- this frame is from a different video. Switch videos above first if you meant to add it there.`);
            return;
        }
        getExportFrame(videoId, n).then((info) => {
            addTrakeEvent({ video_id: videoId, frame_idx: info.frame_idx, thumbnail: info.thumbnail_url });
        }).catch((e) => showStatus(e.message));
    }

    function removeTrakeEvent(i) {
        s.trake.events.splice(i, 1);
        renderEventList();
    }
    function moveTrakeEvent(from, to) {
        const [ev] = s.trake.events.splice(from, 1);
        s.trake.events.splice(to, 0, ev);
        renderEventList();
    }
    function wireTrakeEventDnd(list) {
        for (const row of list.querySelectorAll(".trake-event-row")) {
            const i = Number(row.dataset.index);
            row.addEventListener("dragstart", () => { s.trake.dragIndex = i; });
            row.addEventListener("dragover", (e) => e.preventDefault());
            row.addEventListener("drop", (e) => {
                e.preventDefault();
                if (s.trake.dragIndex === null || s.trake.dragIndex === i) return;
                moveTrakeEvent(s.trake.dragIndex, i);
                s.trake.dragIndex = null;
            });
            const removeBtn = row.querySelector(".export-remove-btn");
            if (removeBtn) removeBtn.onclick = () => removeTrakeEvent(i);
        }
    }

    // Only the event list -- never the video element or the cache panel --
    // so adding/removing/reordering an event never interrupts playback.
    function renderEventList() {
        const list = el("#trake-event-list");
        if (!list) return;
        if (!s.trake.events.length) {
            list.innerHTML = `<div class="status-banner info">No events yet -- play the video and click "Add current frame", or add one from a preview card's "+" / the Frame ID box above.</div>`;
            return;
        }
        list.innerHTML = s.trake.events.map((e, i) => `
            <div class="trake-event-row" draggable="true" data-index="${i}">
                ${e.thumbnail
                    ? `<div class="thumb-wrap thumb-wrap-static"><img src="${e.thumbnail}" loading="lazy"></div>`
                    : `<div class="thumb-missing">no preview</div>`}
                <div class="trake-event-fields">
                    <div class="thumb-caption"><b>E${i + 1}</b></div>
                    <div class="thumb-caption muted">frame ${e.frame_idx}</div>
                </div>
                <button class="icon-btn export-remove-btn" title="Remove" data-index="${i}">✕</button>
            </div>`).join("");
        wireTrakeEventDnd(list);
    }

    // POSTs this video's curated {video_id, frame_idxs} to
    // /api/export/trake-rows and stashes the <=100 returned candidate
    // sequences client-side, keyed by video_id -- repeatable for as many
    // candidate videos as the human wants to compare (each just adds/
    // overwrites its own cache entry, see module docstring). Newly cached
    // (or re-cached) videos default to checked-and-appended into the
    // merge priority order, last -- keeps a freshly regenerated video in
    // whatever priority slot it already had instead of bumping it to the
    // front.
    async function generateRowsForCurationVideo() {
        if (!s.trake.videoId || !s.trake.events.length) {
            showStatus("Add at least one event before generating rows.");
            return;
        }
        const btn = el("#trake-generate-btn");
        if (btn) btn.disabled = true;
        try {
            const frameIdxs = s.trake.events.map((e) => e.frame_idx);
            const thumbnails = s.trake.events.map((e) => e.thumbnail);
            const data = await getTrakeRows(s.trake.videoId, frameIdxs, 100);
            s.trake.cache.set(s.trake.videoId, { frameIdxs, thumbnails, rows: data.rows });
            if (!s.trake.mergeOrder.includes(s.trake.videoId)) s.trake.mergeOrder.push(s.trake.videoId);
            s.trake.mergeChecked.add(s.trake.videoId);
            showStatus(`✓ Cached ${data.rows.length} rows for ${s.trake.videoId}. Curate another video, or check it below and Export.`, "info");
            renderCacheList();
        } catch (e) {
            showStatus(e.message);
        } finally {
            if (btn) btn.disabled = false;
        }
    }

    function removeFromCache(videoId) {
        s.trake.cache.delete(videoId);
        s.trake.mergeOrder = s.trake.mergeOrder.filter((v) => v !== videoId);
        s.trake.mergeChecked.delete(videoId);
        renderCacheList();
    }

    function wireCacheDnd(list) {
        for (const row of list.querySelectorAll(".trake-cache-row")) {
            const vid = row.dataset.videoId;
            row.addEventListener("dragstart", () => { s.trake.mergeDragIndex = s.trake.mergeOrder.indexOf(vid); });
            row.addEventListener("dragover", (e) => e.preventDefault());
            row.addEventListener("drop", (e) => {
                e.preventDefault();
                const from = s.trake.mergeDragIndex;
                const to = s.trake.mergeOrder.indexOf(vid);
                if (from === null || from === to) return;
                const [moved] = s.trake.mergeOrder.splice(from, 1);
                s.trake.mergeOrder.splice(to, 0, moved);
                s.trake.mergeDragIndex = null;
                renderCacheList();
            });
        }
    }

    function renderCacheList() {
        const list = el("#trake-cache-list");
        if (!list) return;
        if (!s.trake.cache.size) {
            list.innerHTML = `<div class="status-banner info">Nothing cached yet -- curate a video above, then click "Generate rows".</div>`;
            return;
        }
        list.innerHTML = s.trake.mergeOrder.map((vid) => {
            const entry = s.trake.cache.get(vid);
            if (!entry) return "";
            const checked = s.trake.mergeChecked.has(vid);
            return `<div class="trake-cache-row" draggable="true" data-video-id="${vid}">
                <input type="checkbox" class="trake-cache-check" data-video-id="${vid}" ${checked ? "checked" : ""}>
                <span class="thumb-caption"><b>${vid}</b> <span class="muted">· ${entry.rows.length} rows · ${entry.frameIdxs.length} events</span></span>
                <button class="icon-btn export-remove-btn" title="Remove from cache" data-video-id="${vid}">✕</button>
            </div>`;
        }).join("");
        list.querySelectorAll(".trake-cache-check").forEach((cb) => {
            cb.onchange = () => {
                const vid = cb.dataset.videoId;
                if (cb.checked) s.trake.mergeChecked.add(vid); else s.trake.mergeChecked.delete(vid);
            };
        });
        list.querySelectorAll(".trake-cache-row .export-remove-btn").forEach((btn) => {
            btn.onclick = () => removeFromCache(btn.dataset.videoId);
        });
        wireCacheDnd(list);
    }

    // Client-side only -- "no CSV parsing, no re-reading files" per spec.
    // Each checked video's own row 1 (its curated pick) goes first, in
    // priority order, then row 2/row 3/... round-robin in that same
    // order until the cap is hit or every cached video's rows are spent.
    // This mirrors the rest of the app's export tiers (one clean row per
    // hypothesis first, hedges/fillers after) while keeping the highest-
    // priority video's own pick at rank 1, which is what R@1 rewards.
    function mergeTrakeCache(maxRows) {
        const selected = s.trake.mergeOrder.filter((vid) => s.trake.mergeChecked.has(vid) && s.trake.cache.has(vid));
        const rows = [];
        for (let k = 0; rows.length < maxRows; k++) {
            let any = false;
            for (const vid of selected) {
                const entry = s.trake.cache.get(vid);
                if (k < entry.rows.length) {
                    rows.push({ video_id: vid, frame_idxs: entry.rows[k] });
                    any = true;
                    if (rows.length >= maxRows) break;
                }
            }
            if (!any) break;
        }
        return rows;
    }

    let trakeSkeletonBuilt = false;

    // Builds the panel's static DOM once (the video element and cache
    // list are updated in place afterward, by renderEventList()/
    // renderCacheList(), never rebuilt wholesale -- rebuilding on every
    // state change would tear down and restart the <video> mid-playback).
    function ensureTrakeSkeleton() {
        if (trakeSkeletonBuilt) return;
        trakeSkeletonBuilt = true;
        el("#exp-trake-content").innerHTML = `
            <div class="trake-curate-panel">
              <div class="trake-toprow">
                <input type="text" id="trake-load-video" placeholder="Video ID e.g. L21_V001">
                <button class="btn" id="trake-load-btn" type="button">Load / switch</button>
                <span id="trake-cur-timer" class="playback-timer">--:-- · frame --</span>
                <button class="btn btn-primary" id="trake-add-btn" type="button">+ Add current frame as event</button>
              </div>
              <div class="trake-main-row">
                <div class="trake-video-col" id="trake-video-wrap">
                  <div class="status-banner info">Load a video above, or open this tab from a result card's ★.</div>
                </div>
                <div class="trake-events-col">
                  <div class="thumb-caption muted" style="margin-bottom:0.4rem;">Events, in sequence order -- drag to reorder, ✕ to remove:</div>
                  <div class="trake-event-list" id="trake-event-list"></div>
                  <button class="btn btn-primary" id="trake-generate-btn" type="button" style="margin-top:0.6rem;">Generate rows for this video</button>
                </div>
              </div>
            </div>
            <hr class="divider">
            <div class="trake-cache-panel">
              <div class="thumb-caption" style="margin-bottom:0.4rem;"><b>Cached videos</b> <span class="muted">(check to include in the merged export, drag to set priority order)</span></div>
              <div id="trake-cache-list"></div>
            </div>`;

        el("#trake-load-btn").onclick = () => loadCurationVideo(el("#trake-load-video").value);
        el("#trake-add-btn").onclick = () => {
            if (!s.trake.videoEl) { showStatus("No video loaded to capture a frame from."); return; }
            const video = s.trake.videoEl;
            const frame_idx = Math.round(video.currentTime * s.trake.fps);
            addTrakeEvent({ video_id: s.trake.videoId, frame_idx, thumbnail: captureVideoThumbnail(video) });
        };
        el("#trake-generate-btn").onclick = generateRowsForCurationVideo;
    }

    function renderTrakeContent() {
        ensureTrakeSkeleton();
        renderEventList();
        renderCacheList();
    }

    // --- wiring ---------------------------------------------------------

    el("#exp-segmented").querySelectorAll("button").forEach((btn) => {
        btn.onclick = () => {
            s.queryType = btn.dataset.type;
            renderTypeVisibility();
            renderTrakeContent();
            renderAnswerContent();
            renderPreview(); // replaceable depends on queryType (TRAKE has no single answer frame)
        };
    });
    el("#exp-name").oninput = (e) => { s.name = e.target.value; };
    el("#exp-answer-text").oninput = (e) => { s.answerText = e.target.value; };
    el("#exp-confirmed").onchange = (e) => {
        s.confirmed = e.target.checked;
        renderTypeVisibility();
        renderAnswerContent();
        renderPreview();
        renderTrakeContent();
    };
    el("#exp-nbr-more").onclick = () => { s.neighboursShown += PREVIEW_PAGE; renderPreview(); };
    el("#exp-sim-more").onclick = () => { s.similarsShown += PREVIEW_PAGE; renderPreview(); };
    el("#exp-cancel").onclick = () => onDone("cancel");

    // Typed "Video ID" / "Frame ID" boxes + Change/Add event button.
    // KIS/VQA: same destination as the preview-pick (applyChangedFrame),
    // but reaches an arbitrary frame not necessarily in either preview
    // list -- verifies the frame actually exists (via /api/export/frame,
    // n-based) before applying, so a typo lands as one clear error rather
    // than a broken export.
    // TRAKE: "Frame ID" is a raw native frame number, not a keyframe n,
    // added to whatever video the curation panel below is already on --
    // the (read-only) Video ID box is just a reminder of that, not a
    // second way to pick the video (see the panel's own Load/switch row
    // for that). No backend round-trip needed at all.
    el("#exp-change-btn").onclick = async () => {
        clearStatus();

        if (s.queryType === "TRAKE") {
            if (!s.trake.videoId) {
                showStatus("Load a video in the TRAKE panel below first.");
                return;
            }
            const num = parseFrameIdInput(el("#exp-change-frame").value);
            if (!num) {
                showStatus("Enter a real frame number.");
                return;
            }
            if (addTrakeEvent({ video_id: s.trake.videoId, frame_idx: num })) {
                el("#exp-change-frame").value = "";
            }
            return;
        }

        const videoId = parseVideoIdInput(el("#exp-change-video").value);
        const num = parseFrameIdInput(el("#exp-change-frame").value);
        if (!videoId || !num) {
            showStatus("Enter a valid video id (e.g. L21_V001) and frame id (e.g. 001).");
            return;
        }

        const btn = el("#exp-change-btn");
        btn.disabled = true;
        try {
            await getExportFrame(videoId, num); // throws if that frame doesn't exist for that video
            applyChangedFrame({ video_id: videoId, n: num });
            el("#exp-change-video").value = "";
            el("#exp-change-frame").value = "";
        } catch (e) {
            showStatus(e.message);
        } finally {
            btn.disabled = false;
        }
    };

    el("#exp-export").onclick = async () => {
        clearStatus();

        if (!s.name) {
            showStatus("Enter a query number.");
            return;
        }

        if (s.queryType === "TRAKE") {
            // Client-side merge of the per-video cache -- no candidates/
            // confirmed/answers body to build, unlike KIS/VQA below.
            const merged = mergeTrakeCache(100);
            if (!merged.length) {
                showStatus("Nothing to export -- curate a video, click \"Generate rows\", then check it below.");
                return;
            }
            const filename = queryFilename("TRAKE", s.name);
            const exportBtn = el("#exp-export");
            exportBtn.disabled = true;
            try {
                await writeTrakeCsv(merged, filename);
                showStatus(`✓ Exported ${filename}.csv (${merged.length} rows) -- you can export again from here if needed.`, "info");
            } catch (e) {
                showStatus(e.message);
            } finally {
                exportBtn.disabled = false;
            }
            return;
        }

        if (s.confirmed && !s.answerFrame) {
            showStatus("No confirmed frame -- open this from a result card.");
            return;
        }
        if (!s.confirmed && !s.answers.length) {
            showStatus("Add at least one answer frame from the preview.");
            return;
        }
        const body = {
            query_type: s.queryType,
            mode: s.confirmed ? "confirmed" : "unconfirmed",
            candidates: getCandidates(),
            confirmed: s.confirmed ? s.answerFrame : null,
            answers: s.confirmed ? [] : s.answers,
            answer: s.answerText,
            neighbour_count: NEIGHBOUR_COUNT_EXPORT,
            filename: queryFilename(s.queryType, s.name),
        };

        const exportBtn = el("#exp-export");
        exportBtn.disabled = true;
        try {
            await exportCsv(body);
            // Left in place, not closed/torn down -- the form stays exactly
            // as it was, so the user can immediately re-export (a new query
            // #, a tweaked frame, ...) without reopening this tab.
            showStatus(`✓ Exported ${body.filename}.csv -- you can export again from here if needed.`, "info");
        } catch (e) {
            showStatus(e.message);
        } finally {
            exportBtn.disabled = false;
        }
    };

    // A "frame" trigger (from video playback) has no keyframe n at all --
    // KIS/VQA's flat CSV path needs one (backend's frame_idx_for_n), so
    // those two types aren't usable here. TRAKE stays the only option.
    if (trigger.kind === "frame") {
        for (const type of ["KIS", "VQA"]) {
            const btn = el(`#exp-segmented button[data-type="${type}"]`);
            btn.disabled = true;
            btn.title = "Needs a keyframe-backed frame -- use TRAKE for a native frame from playback.";
        }
    }

    renderTypeVisibility();
    renderAnswerContent();
    renderPreview();
    renderTrakeContent();

    // Seed the curation panel straight from `trigger` when it carries a
    // video/frame of its own -- any signal's result card, a real TRAKE
    // candidate's own matched events, or a raw playback frame can all
    // start (or extend) a TRAKE sequence, not just a real TRAKE search
    // (see module docstring). Not limited to when queryType actually
    // starts on TRAKE -- switching to TRAKE later still finds the panel
    // already seeded.
    // loadCurationVideo is called right after kicking the event-seeding
    // off (not awaited first) so the video starts loading immediately
    // rather than waiting on the frame-info round trip(s) below; its own
    // videoId!==s.trake.videoId check still resets s.trake.events first,
    // but that's a no-op here since freshState() always starts empty.
    let seedVideoId = null;
    if (trigger.kind === "trake") {
        // Resolved in parallel but applied in original event order (matters
        // here, unlike a lone addTrakeEventFromN call elsewhere) -- several
        // concurrent fetches racing straight into addTrakeEvent could
        // otherwise land E2 before E1 depending on which response arrives
        // first.
        seedVideoId = trigger.candidate.video_id;
        const matched = trigger.candidate.events.filter((e) => e.matched);
        Promise.all(matched.map((e) => getExportFrame(seedVideoId, e.n).catch(() => null))).then((infos) => {
            for (const info of infos) {
                if (info) addTrakeEvent({ video_id: seedVideoId, frame_idx: info.frame_idx, thumbnail: info.thumbnail_url });
            }
        });
    } else if (trigger.kind === "flat") {
        seedVideoId = trigger.video_id;
        addTrakeEventFromN(trigger.video_id, trigger.n);
    } else if (trigger.kind === "frame") {
        seedVideoId = trigger.video_id;
        addTrakeEvent({ video_id: trigger.video_id, frame_idx: trigger.frame_idx });
    }
    if (seedVideoId) loadCurationVideo(seedVideoId);
}
