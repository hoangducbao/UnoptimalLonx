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
//          /api/export/trake-rows and caches the <=99 returned candidate
//          sequences client-side, keyed by video_id (`s.trake.cache`
//          below) -- repeatable for as many candidate videos as the human
//          wants to compare, each just adding another entry.
//       3. Export: the human checks which cached videos to include and
//          their priority order; the rows are interleaved client-side (no
//          backend round-trip, no re-reading anything) into one <=99-row
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
//   {kind: "frame", video_id, frame_idx, current_time?, thumbnail?} -- a raw
//     native frame from a video playback dialog, no keyframe n at all
//     (TRAKE-only: KIS/VQA need an n for the backend's flat CSV path, which
//     this trigger doesn't have). current_time/thumbnail are optional --
//     dialogs.js's "Export this frame" buttons send both (the playback
//     dialog's own currentTime + a canvas snapshot) so the curation video
//     below resumes from the same spot and the seeded event gets a real
//     preview; a trigger without them still works, just starts the
//     curation video at 0:00 with no event thumbnail, as before.

import { exportCsv, getExportFrame, getExportNearestKeyframe, getExportNeighbors, getExportSimilar, getPlayback, getTrakeRows, writeTrakeCsv } from "./api.js";
import { fmtTime } from "./format.js";
import { applyVideoPrefs, bindSpeedShortcut, captureVideoThumbnail } from "./video-controls.js";
import { tile } from "./settings.js";

const NEIGHBOUR_COUNT_EXPORT = 10; // fixed row-generation window, independent of preview expand state

// Initial count and "Show N more" step per preview section -- three rows of
// whatever the tile-size setting makes .export-preview-grid wide (settings.js's
// TILE_SIZES.previewPage/previewColumns), read fresh rather than captured at
// import so a size change in the search tab reaches this one (see settings.js's
// storage listener).
const previewPage = () => tile().previewPage;

function freshState(trigger) {
    const isFlat = trigger.kind === "flat";
    const seed = isFlat ? { video_id: trigger.video_id, n: trigger.n } : null;

    return {
        trigger,
        queryType: isFlat ? "KIS" : "TRAKE", // "trake"/"frame" triggers default to TRAKE -- "frame" has no n for KIS/VQA's flat CSV path
        name: "",
        confirmed: true,
        // Export tab's "Keyframes" checkbox (KIS/VQA only): checked (old
        // behavior, unchanged) means the answer is an indexed keyframe n,
        // same as everything below. Unchecked means it's a raw native
        // frame_idx instead, curated via the `native` video-playback panel
        // below rather than the Neighbours/Similars grids -- see the
        // "KIS/VQA native (Keyframes-unchecked) curation" section further
        // down. Defaults unchecked only for a "frame" trigger (opened from
        // video playback, item 7), which has no keyframe n to check it
        // *to* in the first place; togglable regardless, snapping to the
        // nearest keyframe on re-check (getExportNearestKeyframe).
        keyframes: trigger.kind !== "frame",
        answerText: "",
        answerFrame: seed,          // confirmed-mode single answer (KIS/VQA, keyframe n space)
        answers: seed ? [seed] : [], // unconfirmed-mode ordered list, pre-seeded per spec ("at least 1 frame")
        frameInfo: new Map(),        // "vid|n" -> {frame_idx, thumbnail_url} | "pending"
        neighbourFrames: null,       // cached /api/export/neighbors result (grows with `neighboursShown`)
        neighboursShown: previewPage(),
        similarFrames: null,         // confirmed mode: cached /api/export/similar result, keyed to similarFramesKey below
        similarFramesKey: null,      // frameKey(answerFrame) similarFrames was fetched for -- refetch when the confirmed frame changes
        similarsShown: previewPage(),
        dragIndex: null,
        // TRAKE: no confirmed/unconfirmed distinction any more -- one
        // per-video curation session (video + ordered event list, each a
        // native frame_idx with its own add-time thumbnail) feeds a
        // "Generate rows" call whose <=99 candidate sequences are cached
        // here per video_id; a final merge step interleaves however many
        // cached videos the human picked, in priority order, into one CSV.
        // See freshCurationState()/freshTrakeState() below and the
        // "TRAKE: curate one video's events..." section further down for
        // the rest.
        trake: freshTrakeState(),
        // KIS/VQA native (Keyframes unchecked): the same per-video
        // curation session shape as `trake` above (video + ordered
        // frame_idx list), minus the cache/merge fields -- one "Generate
        // rows" video isn't a thing here, the CSV is built straight from
        // whatever's curated (see the export handler's native branch).
        // Confirmed mode caps this list at one entry (replace, not
        // append); unconfirmed allows several, same temporal-order insert
        // as TRAKE events. See addEventToCuration() below.
        native: freshCurationState(),
    };
}

// Shared shape for both TRAKE's per-video curation session and KIS/VQA's
// native (Keyframes-unchecked) one -- `trake` uses every field below plus
// its own cache/merge fields (freshTrakeState() further down layers those
// on top); `native` uses this as-is.
function freshCurationState() {
    return {
        videoId: null,
        events: [],       // [{frame_idx, thumbnail}] -- TRAKE keeps this in temporal (frame_idx) order; native unconfirmed mode is insertion order instead (see addEventToCuration)
        dragIndex: null,
        videoEl: null,     // the curation panel's live <video>, for Add/capture
        unbindSpeedShortcut: null, // bindSpeedShortcut()'s cleanup for the current videoEl
        fps: 25,
    };
}

function freshTrakeState() {
    return {
        ...freshCurationState(),
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

// AIC submission naming: query-p3-<#>-<type>.csv, matching the real
// submission/*.csv samples in the repo -- note VQA's type slug is "qa",
// not "vqa".
const TYPE_SLUG = { KIS: "kis", VQA: "qa", TRAKE: "trake" };
function queryFilename(queryType, name) {
    return `query-p3-${name}-${TYPE_SLUG[queryType]}`;
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
          <input type="number" id="exp-name" min="1" step="1" placeholder="QUERY NUMBER - REQUIRED">
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
          <input type="checkbox" id="exp-keyframes" style="margin:0 0 0 1rem;">
          <label for="exp-keyframes" style="margin:0;">Keyframes</label>
        </div>
        <div id="exp-flat-body">
          <div class="export-answer-area" id="exp-answer-area">
            <div id="exp-answer-content"></div>
            <input type="text" id="exp-answer-text" placeholder="VQA answer" style="display:none;">
          </div>
          <div id="exp-native-content" style="display:none;"></div>
        </div>
        <div id="exp-trake-body" style="display:none;">
          <div id="exp-trake-content"></div>
        </div>
        <div class="export-preview-area" id="exp-preview-area">
          <div class="export-preview-section">
            <div class="export-preview-header"><b>Neighbours</b> <span class="thumb-caption muted">(nearest keyframes by time)</span></div>
            <div class="grid export-preview-grid" id="exp-nbr-grid"></div>
            <button class="btn" id="exp-nbr-more">Show ${previewPage()} more</button>
          </div>
          <div class="export-preview-section">
            <div class="export-preview-header"><b>Similars</b> <span class="thumb-caption muted" id="exp-sim-caption">(this query's ranked results)</span></div>
            <div class="grid export-preview-grid" id="exp-sim-grid"></div>
            <button class="btn" id="exp-sim-more">Show ${previewPage()} more</button>
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
        // KIS/VQA with "Keyframes" unchecked: curated via a video-playback
        // panel (like TRAKE's) instead of the keyframe grids, in native
        // frame_idx space -- see the "KIS/VQA native curation" section.
        const nativeMode = !isTrake && !s.keyframes;
        const isVqa = s.queryType === "VQA";
        el("#exp-trake-body").style.display = isTrake ? "block" : "none";
        el("#exp-flat-body").style.display = isTrake ? "none" : "block";
        // No confirmed/unconfirmed distinction for TRAKE any more (see
        // module docstring) -- the row (Confirmed + Keyframes) only means
        // something for KIS/VQA.
        el("#exp-confirmed-row").style.display = isTrake ? "none" : "flex";
        // #exp-answer-text gets relocated out of #exp-answer-area below
        // (native+VQA only, into #exp-change-fields) -- hiding the whole
        // area here doesn't take it down too, since a moved DOM node is
        // no longer a descendant of its old (now-hidden) parent.
        el("#exp-answer-area").style.display = nativeMode ? "none" : "block";
        el("#exp-native-content").style.display = nativeMode ? "block" : "none";
        // No Neighbours/Similars preview for TRAKE, or for native KIS/VQA
        // (Keyframes unchecked) -- per spec, neither has a keyframe-space
        // "similar" pool to browse (row generation computes the
        // equivalent server-side, snapped to the nearest keyframe for
        // native mode -- see backend/export.py's
        // similar_candidates_for_native_frame). renderPreview() itself
        // skips fetching for both cases too, not just this toggle.
        el("#exp-preview-area").style.display = (isTrake || nativeMode) ? "none" : "flex";

        // Video ID/Frame ID/Change-or-Add-event: TRAKE still repurposes
        // this row (a raw frame_idx event, added to whatever video the
        // TRAKE panel below is curating) and keyframe-mode KIS/VQA keeps
        // its original "Change" lookup -- native (Keyframes-unchecked)
        // KIS/VQA drops the Video ID/Frame ID/button trio entirely
        // instead (item 1.1, both KIS and VQA): the only way to add a
        // frame there is the curation panel's own video. VQA reuses that
        // vacated topbar slot for the answer text box (item 2, "export
        // this qa answer same as checked keyframes mode" -- same
        // s.answerText field either way, just needs somewhere visible to
        // type into once #exp-answer-area itself is out of the picture);
        // KIS leaves the whole row empty/hidden.
        const changeFields = el("#exp-change-fields");
        const changeVideo = el("#exp-change-video");
        const changeFrame = el("#exp-change-frame");
        const changeBtn = el("#exp-change-btn");
        const answerText = el("#exp-answer-text");
        if (nativeMode) {
            changeVideo.style.display = "none";
            changeFrame.style.display = "none";
            changeBtn.style.display = "none";
            changeFields.style.display = isVqa ? "flex" : "none";
            if (isVqa) changeFields.append(answerText); // .export-change-fields input styles it to fill the row
        } else {
            changeVideo.style.display = "block";
            changeFrame.style.display = "block";
            changeBtn.style.display = "block";
            changeFields.style.display = "flex";
            el("#exp-answer-area").append(answerText); // back to its normal spot, after #exp-answer-content
            changeFrame.placeholder = isTrake ? "Frame ID (real frame)" : "Frame ID";
            changeBtn.textContent = isTrake ? "Add event" : "Change";
            changeVideo.readOnly = isTrake;
            changeVideo.value = isTrake ? (s.trake.videoId || "") : "";
            changeVideo.title = isTrake ? "Switch curation video from the panel below" : "";
        }

        // Same typing box either way -- unconfirmed mode used to disable
        // this with an "LLM needed" placeholder (answering unconfirmed
        // VQA queries was meant to be automated later), but that's no
        // longer the plan: a human types the answer regardless of mode.
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
        // Native (Keyframes-unchecked) KIS/VQA uses the curation panel in
        // #exp-native-content instead (hidden along with this whole area,
        // see renderTypeVisibility) -- nothing to fetch/render here, and
        // skipping avoids a wasted ensureFrameInfo() round trip for a
        // stale s.answerFrame/s.answers left over from keyframe mode.
        if (!s.keyframes) {
            content.innerHTML = "";
            return;
        }
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
        // No Neighbours/Similars preview for TRAKE, or for native
        // (Keyframes-unchecked) KIS/VQA -- see renderTypeVisibility's
        // #exp-preview-area toggle and module docstring.
        if (s.queryType === "TRAKE" || !s.keyframes) return;

        const addable = !s.confirmed;
        const replaceable = s.confirmed;

        // Re-labelled every render, not just at build time: the tile-size
        // setting these counts come from can change in the search tab while
        // this one is open (settings.js's storage listener), and the buttons'
        // own handlers already step by the current previewPage().
        el("#exp-nbr-more").textContent = `Show ${previewPage()} more`;
        el("#exp-sim-more").textContent = `Show ${previewPage()} more`;

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

        // Similars. Confirmed mode: a fresh visual search seeded by the
        // confirmed frame itself (getExportSimilar, see backend/export.py's
        // similar_candidates_for_frame) -- "similar to the picked image",
        // not whatever the opener tab's last query happened to find.
        // Unconfirmed mode has no single confirmed frame to re-query from,
        // so it still shows the opener tab's own already-fetched, already-
        // ranked results (getCandidates()); those may be TRAKE-shaped
        // ({video_id, events}, no .n) if the opener tab's last search was a
        // real TRAKE search -- previewCardHtml needs a flat {video_id, n}
        // shape, so fall back to a plain message rather than rendering
        // broken cards.
        const simGrid = el("#exp-sim-grid");
        el("#exp-sim-caption").textContent = replaceable ? "(visual search from the confirmed frame)" : "(this query's ranked results)";
        if (replaceable) {
            if (!s.answerFrame) {
                simGrid.innerHTML = `<div class="status-banner info">No confirmed frame to search similar images from.</div>`;
                el("#exp-sim-more").style.display = "none";
            } else {
                const key = frameKey(s.answerFrame);
                if (s.similarFramesKey !== key || !s.similarFrames || s.similarFrames.length < s.similarsShown) {
                    simGrid.innerHTML = `<div class="status-banner info">Loading…</div>`;
                    try {
                        const data = await getExportSimilar(s.answerFrame.video_id, s.answerFrame.n, s.similarsShown);
                        s.similarFrames = data.results;
                        s.similarFramesKey = key;
                    } catch (e) {
                        simGrid.innerHTML = `<div class="status-banner error">${e.message}</div>`;
                        return;
                    }
                }
                const similars = s.similarFrames.slice(0, s.similarsShown);
                simGrid.innerHTML = similars.map((c) => previewCardHtml(c, { addable, replaceable }))
                    .join("") || `<div class="status-banner info">No similar frames found.</div>`;
                el("#exp-sim-more").style.display = s.similarFrames.length >= s.similarsShown ? "block" : "none";
            }
        } else {
            const candidates = getCandidates();
            if (candidates.length && !("n" in candidates[0])) {
                simGrid.innerHTML = `<div class="status-banner info">Last search wasn't a flat-result signal -- no Similars to preview.</div>`;
                el("#exp-sim-more").style.display = "none";
            } else {
                const similars = candidates.slice(0, s.similarsShown);
                simGrid.innerHTML = similars.map((c) => previewCardHtml(c, { addable, replaceable }))
                    .join("") || `<div class="status-banner info">No results from the last search.</div>`;
                el("#exp-sim-more").style.display = candidates.length > s.similarsShown ? "block" : "none";
            }
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

    // --- Shared curation-panel machinery: TRAKE's per-video event list
    // (curate -> cache its generated rows -> merge several cached videos)
    // and KIS/VQA's native (Keyframes-unchecked) answer-frame curation
    // both boil down to "one video, an ordered native frame_idx list,
    // played and captured from a live <video>" -- `kind` ("trake" |
    // "native") picks which state bucket (s.trake / s.native) and DOM ids
    // a call operates on. TRAKE layers its own cache/merge panel on top
    // (generateRowsForCurationVideo() etc. below, unaffected by this) --
    // "native" has no such extra step, the curated list goes straight
    // into the export payload. ---------------------------------------

    function curationTarget(kind) {
        return kind === "trake"
            ? { state: s.trake, wrapId: "#trake-video-wrap", timerId: "#trake-cur-timer", speedId: "#trake-cur-speed", listId: "#trake-event-list" }
            : { state: s.native, wrapId: "#native-video-wrap", timerId: "#native-cur-timer", speedId: "#native-cur-speed", listId: "#native-event-list" };
    }

    // Loads (or switches the curation panel to) a video: fetches playback
    // info and builds a fresh <video>. Switching to a *different* video
    // than the one currently being curated starts a clean event list --
    // any TRAKE cache entry already generated for either video is
    // untouched (cache entries persist independently of what's in the
    // live curation panel, see generateRowsForCurationVideo()).
    // `seekTime` (seconds) picks up video playback exactly where the user
    // left it in whatever dialog they hit "Export this frame" from (see
    // dialogs.js's current_time handoff). `seekN` is the alternative for a
    // keyframe-backed frame (a KIS/VQA confirmed/answer frame, switched to
    // TRAKE, or a "flat" trigger's own seed): rather than convert n to a
    // time ourselves, /api/playback already resolves a keyframe's own
    // timestamp server-side (same lookup a KIS/VQA playback dialog uses),
    // so passing it as `n` here gets the exact same start_time. `seekFrameIdx`
    // is the native-space equivalent, for handing a raw frame_idx (no n)
    // off between the two curation panels -- computed from the freshly
    // fetched fps once it's known. None given (all default) starts at
    // 0:00, same as before, for every other way a panel gets loaded/
    // switched (typed Video ID, a plain video switch, a "trake" trigger's
    // own multi-event seeding).
    async function loadCurationVideo(videoId, kind, { seekTime = 0, seekN = null, seekFrameIdx = null } = {}) {
        const { state, wrapId, timerId, speedId } = curationTarget(kind);
        videoId = (videoId || "").trim().toUpperCase();
        if (!videoId) return;
        if (videoId !== state.videoId) {
            state.videoId = videoId;
            state.events = [];
            renderCurationEventList(kind);
        }
        clearStatus();
        renderTypeVisibility(); // keeps the topbar's locked Video ID display in sync
        const wrap = el(wrapId);
        if (wrap) wrap.innerHTML = `<div class="status-banner info">Loading…</div>`;
        try {
            const data = await getPlayback(videoId, seekN ?? undefined);
            if (el(wrapId) !== wrap) return; // panel torn down mid-fetch (query type switched away)
            state.fps = data.fps;
            wrap.innerHTML = "";
            const video = document.createElement("video");
            let t = seekTime;
            if (seekN != null) t = data.start_time;
            else if (seekFrameIdx != null) t = seekFrameIdx / (data.fps || 25);
            video.src = data.video_url + (t > 0 ? `#t=${t}` : "");
            video.controls = true;
            wrap.append(video);
            applyVideoPrefs(video);
            if (state.unbindSpeedShortcut) state.unbindSpeedShortcut(); // drop the outgoing <video>'s listener before binding the new one
            state.unbindSpeedShortcut = bindSpeedShortcut(video, wrap, el(speedId));
            state.videoEl = video;
            const timer = el(timerId);
            video.addEventListener("timeupdate", () => {
                timer.textContent = `${fmtTime(video.currentTime)} · frame ${Math.round(video.currentTime * state.fps)}`;
            });
        } catch (e) {
            state.videoEl = null;
            if (wrap) wrap.innerHTML = `<div class="status-banner error">${e.message}</div>`;
        }
    }

    // Adds one event to the video currently being curated -- from the
    // inline "Add current frame" button (f.thumbnail already captured) or
    // the repurposed Frame ID/Change row (raw frame_idx, no thumbnail).
    // Enforces the one hard constraint: a TRAKE export row (or a native
    // KIS/VQA answer list) is exactly one video, so a frame from a
    // different video is rejected rather than silently starting a second,
    // unrepresentable sequence.
    //
    // TRAKE inserts in temporal order (by frame_idx) rather than always
    // appending -- adding an earlier frame after later ones (e.g.
    // scrubbing back, or the Frame ID box) lands it in the right spot
    // immediately instead of needing a manual drag to fix the sequence.
    // Assumes the list is already ordered, which holds as long as every
    // addition goes through here; drag-to-reorder can still freely
    // override this after the fact.
    // Native (KIS/VQA, Keyframes unchecked) is different in kind, not
    // just order: confirmed mode is exactly one answer frame, so a new
    // pick there replaces the list instead of inserting into it (its "one
    // frame, whatever's live" caption doesn't have a rank to preserve);
    // unconfirmed mode is a plain append, no temporal sort at all -- each
    // pick becomes the next "Cand N" in whatever order the human added
    // them, not sorted by frame_idx (item 1.3).
    function addEventToCuration(kind, f) {
        const { state } = curationTarget(kind);
        if (state.videoId && f.video_id !== state.videoId) {
            showStatus(`Currently curating ${state.videoId} -- this frame is from a different video. Switch videos above first if you meant to add it there.`);
            return false;
        }
        if (!state.videoId) state.videoId = f.video_id;
        const entry = { frame_idx: f.frame_idx, thumbnail: f.thumbnail ?? null };
        if (kind === "native") {
            if (s.confirmed) state.events = [entry];
            else state.events.push(entry);
        } else {
            const insertAt = state.events.findIndex((e) => e.frame_idx > entry.frame_idx);
            if (insertAt === -1) state.events.push(entry);
            else state.events.splice(insertAt, 0, entry);
        }
        clearStatus();
        renderCurationEventList(kind);
        return true;
    }

    // Resolves a keyframe n (not a raw frame_idx) to its real frame_idx +
    // thumbnail before adding -- used to seed a curation panel from a
    // "flat" trigger (any non-TRAKE signal's ★, which only carries n, no
    // frame_idx) via the same lookup the rest of the app already does.
    function addEventFromN(kind, videoId, n) {
        const { state } = curationTarget(kind);
        if (state.videoId && videoId !== state.videoId) {
            showStatus(`Currently curating ${state.videoId} -- this frame is from a different video. Switch videos above first if you meant to add it there.`);
            return;
        }
        getExportFrame(videoId, n).then((info) => {
            addEventToCuration(kind, { video_id: videoId, frame_idx: info.frame_idx, thumbnail: info.thumbnail_url });
        }).catch((e) => showStatus(e.message));
    }

    function removeCurationEvent(kind, i) {
        const { state } = curationTarget(kind);
        state.events.splice(i, 1);
        renderCurationEventList(kind);
    }
    function moveCurationEvent(kind, from, to) {
        const { state } = curationTarget(kind);
        const [ev] = state.events.splice(from, 1);
        state.events.splice(to, 0, ev);
        renderCurationEventList(kind);
    }
    function wireCurationEventDnd(kind, list) {
        const { state } = curationTarget(kind);
        for (const row of list.querySelectorAll(".trake-event-row")) {
            const i = Number(row.dataset.index);
            row.addEventListener("dragstart", () => { state.dragIndex = i; });
            row.addEventListener("dragover", (e) => e.preventDefault());
            row.addEventListener("drop", (e) => {
                e.preventDefault();
                if (state.dragIndex === null || state.dragIndex === i) return;
                moveCurationEvent(kind, state.dragIndex, i);
                state.dragIndex = null;
            });
            const removeBtn = row.querySelector(".export-remove-btn");
            if (removeBtn) removeBtn.onclick = () => removeCurationEvent(kind, i);
        }
    }

    // Only the event list -- never the video element or the TRAKE cache
    // panel -- so adding/removing/reordering an event never interrupts
    // playback.
    //
    // Caption differs by kind/mode (items 1.2/1.3): TRAKE keeps "E1, E2,
    // ..." (a sequence of events within one video). Native confirmed mode
    // is exactly one frame with no rank to show, so its label is the
    // video_id instead of "E1". Native unconfirmed mode is a pool of
    // candidate answer frames, not a sequence -- "Cand 1, Cand 2, ..."
    // instead of "E1, E2, ...", with the video_id shown too (all
    // candidates share state.videoId, same one-video-per-curation-session
    // constraint as TRAKE, but it's less obviously implied here since
    // there's no "sequence" framing to carry it).
    function renderCurationEventList(kind) {
        const { state, listId } = curationTarget(kind);
        const list = el(listId);
        if (!list) return;
        if (!state.events.length) {
            const msg = kind === "trake"
                ? `No events yet -- play the video and click "Add current frame as event", or add one via the Frame ID box above.`
                : s.confirmed
                    ? `No frame chosen yet -- play the video and click "Switch to this frame".`
                    : `No candidates yet -- play the video and click "+ Add current frame".`;
            list.innerHTML = `<div class="status-banner info">${msg}</div>`;
            return;
        }
        list.innerHTML = state.events.map((e, i) => {
            const caption = kind === "trake"
                ? `<b>E${i + 1}</b> <span class="muted">· frame ${e.frame_idx}</span>`
                : s.confirmed
                    ? `<b>${state.videoId}</b> <span class="muted">· frame ${e.frame_idx}</span>`
                    : `<b>Cand ${i + 1}</b> <span class="muted">· ${state.videoId} · frame ${e.frame_idx}</span>`;
            return `
            <div class="trake-event-row" draggable="true" data-index="${i}">
                ${e.thumbnail
                    ? `<div class="thumb-wrap thumb-wrap-static"><img src="${e.thumbnail}" loading="lazy"></div>`
                    : `<div class="thumb-missing">no preview</div>`}
                <div class="trake-event-fields">
                    <div class="thumb-caption">${caption}</div>
                </div>
                <button class="icon-btn export-remove-btn" title="Remove" data-index="${i}">✕</button>
            </div>`;
        }).join("");
        wireCurationEventDnd(kind, list);
    }

    // POSTs this video's curated {video_id, frame_idxs} to
    // /api/export/trake-rows and stashes the <=99 returned candidate
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
            const data = await getTrakeRows(s.trake.videoId, frameIdxs, 99);
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
    // list are updated in place afterward, by renderCurationEventList()/
    // renderCacheList(), never rebuilt wholesale -- rebuilding on every
    // state change would tear down and restart the <video> mid-playback).
    function ensureTrakeSkeleton() {
        if (trakeSkeletonBuilt) return;
        trakeSkeletonBuilt = true;
        el("#exp-trake-content").innerHTML = `
            <div class="trake-curate-panel">
              <div class="trake-toprow">
                <input type="text" id="trake-load-video" class="curate-load-video" placeholder="Video ID e.g. L21_V001">
                <button class="btn curate-load-btn" id="trake-load-btn" type="button">Load / switch</button>
                <span id="trake-cur-timer" class="playback-timer">--:-- · frame --</span>
                <span id="trake-cur-speed" class="playback-speed" title="&lt; / , slower, &gt; / . faster, 0 resets to 1x">1x</span>
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

        el("#trake-load-btn").onclick = () => loadCurationVideo(el("#trake-load-video").value, "trake");
        el("#trake-add-btn").onclick = () => {
            if (!s.trake.videoEl) { showStatus("No video loaded to capture a frame from."); return; }
            const video = s.trake.videoEl;
            const frame_idx = Math.round(video.currentTime * s.trake.fps);
            addEventToCuration("trake", { video_id: s.trake.videoId, frame_idx, thumbnail: captureVideoThumbnail(video) });
        };
        el("#trake-generate-btn").onclick = generateRowsForCurationVideo;
    }

    function renderTrakeContent() {
        ensureTrakeSkeleton();
        renderCurationEventList("trake");
        renderCacheList();
    }

    // --- KIS/VQA native (Keyframes-unchecked) curation: same video-
    // playback panel as TRAKE's, minus the Generate-rows/cache/merge step
    // -- the curated list (capped at one frame in confirmed mode) goes
    // straight into the export payload, see the export handler's native
    // branch below. ------------------------------------------------------

    let nativeSkeletonBuilt = false;

    function ensureNativeSkeleton() {
        if (nativeSkeletonBuilt) return;
        nativeSkeletonBuilt = true;
        el("#exp-native-content").innerHTML = `
            <div class="trake-curate-panel">
              <div class="trake-toprow">
                <input type="text" id="native-load-video" class="curate-load-video" placeholder="Video ID e.g. L21_V001">
                <button class="btn curate-load-btn" id="native-load-btn" type="button">Load / switch</button>
                <span id="native-cur-timer" class="playback-timer">--:-- · frame --</span>
                <span id="native-cur-speed" class="playback-speed" title="&lt; / , slower, &gt; / . faster, 0 resets to 1x">1x</span>
                <button class="btn btn-primary" id="native-add-btn" type="button">+ Add current frame</button>
              </div>
              <div class="trake-main-row">
                <div class="trake-video-col" id="native-video-wrap">
                  <div class="status-banner info">Load a video above, or open this tab from a result card's ★.</div>
                </div>
                <div class="trake-events-col">
                  <div class="thumb-caption muted" id="native-events-label" style="margin-bottom:0.4rem;"></div>
                  <div class="trake-event-list" id="native-event-list"></div>
                </div>
              </div>
            </div>`;

        el("#native-load-btn").onclick = () => loadCurationVideo(el("#native-load-video").value, "native");
        el("#native-add-btn").onclick = () => {
            if (!s.native.videoEl) { showStatus("No video loaded to capture a frame from."); return; }
            const video = s.native.videoEl;
            const frame_idx = Math.round(video.currentTime * s.native.fps);
            addEventToCuration("native", { video_id: s.native.videoId, frame_idx, thumbnail: captureVideoThumbnail(video) });
        };
    }

    function renderNativeContent() {
        ensureNativeSkeleton();
        const label = el("#native-events-label");
        if (label) {
            label.textContent = s.confirmed
                ? "Chosen frame:"
                : "Candidate answer frames, in the order added -- drag to reorder, ✕ to remove:";
        }
        // Confirmed mode is exactly one answer frame -- clicking Add again
        // doesn't add a second one, it swaps to whatever's playing now
        // (item 1.2), so the button reads that way instead of "Add".
        // Unconfirmed keeps the plain "add" phrasing (item 1.3 only asked
        // to rename the confirmed-mode button).
        const addBtn = el("#native-add-btn");
        if (addBtn) addBtn.textContent = s.confirmed ? "Switch to this frame" : "+ Add current frame";
        renderCurationEventList("native");
    }

    // --- wiring ---------------------------------------------------------

    el("#exp-segmented").querySelectorAll("button").forEach((btn) => {
        btn.onclick = () => {
            // Snapshot before s.queryType flips -- whichever KIS/VQA frame
            // was in play (confirmed mode's single answerFrame, or
            // unconfirmed's first pick) is "the chosen keyframe" a switch
            // into TRAKE should match playback to, so the curation video
            // resumes at that exact keyframe's timestamp instead of 0:00.
            const enteringTrake = btn.dataset.type === "TRAKE" && s.queryType !== "TRAKE";
            const seedFrame = enteringTrake ? (s.answerFrame || s.answers[0] || null) : null;
            // Reverse handoff: leaving TRAKE into a Keyframes-unchecked
            // KIS/VQA whose native panel hasn't loaded a video yet (e.g.
            // seeded straight from a "frame" trigger -- item 7, both
            // panels get seeded with the same raw frame up front, see the
            // bottom of this function) -- match its playback to wherever
            // that seed left off, same idea as seedFrame above but the
            // other direction.
            const enteringNative = btn.dataset.type !== "TRAKE" && s.queryType === "TRAKE" && !s.keyframes
                && s.native.videoId && !s.native.videoEl;
            const nativeSeekFrame = enteringNative ? s.native.events[0] : null;
            s.queryType = btn.dataset.type;
            renderTypeVisibility();
            renderTrakeContent();
            renderNativeContent();
            renderAnswerContent();
            renderPreview(); // replaceable depends on queryType (TRAKE has no single answer frame)
            if (seedFrame) loadCurationVideo(seedFrame.video_id, "trake", { seekN: seedFrame.n });
            if (enteringNative) {
                loadCurationVideo(s.native.videoId, "native", nativeSeekFrame ? { seekFrameIdx: nativeSeekFrame.frame_idx } : {});
            }
        };
    });
    el("#exp-name").oninput = (e) => { s.name = e.target.value; };
    el("#exp-answer-text").oninput = (e) => { s.answerText = e.target.value; };
    el("#exp-confirmed").onchange = (e) => {
        s.confirmed = e.target.checked;
        // Native mode's list is shared storage between confirmed (exactly
        // one frame) and unconfirmed (several) -- collapse down to the
        // first entry on the way into confirmed, same idea as
        // applyChangedFrame's n-space equivalent below.
        if (s.confirmed && s.native.events.length > 1) s.native.events = [s.native.events[0]];
        renderTypeVisibility();
        renderAnswerContent();
        renderNativeContent();
        renderPreview();
        renderTrakeContent();
    };
    el("#exp-keyframes").checked = s.keyframes;
    el("#exp-keyframes").onchange = async (e) => {
        const next = e.target.checked;
        clearStatus();
        e.target.disabled = true;
        s.keyframes = next;
        renderTypeVisibility();
        renderNativeContent(); // ensures the native skeleton exists before loadCurationVideo below touches it
        renderAnswerContent();
        renderPreview();
        try {
            if (next && s.native.events.length) {
                // Re-checking: snap whatever native frame(s) are curated to
                // their nearest keyframe n -- best-effort per event, since a
                // raw playback frame rarely lands exactly on one.
                const resolved = await Promise.all(s.native.events.map((ev) =>
                    getExportNearestKeyframe(s.native.videoId, ev.frame_idx).catch(() => null)));
                const frames = resolved.filter(Boolean).map((r) => ({ video_id: s.native.videoId, n: r.n }));
                if (frames.length) {
                    s.answerFrame = frames[0];
                    s.answers = frames;
                    renderAnswerContent();
                    renderPreview();
                }
            } else if (!next && !s.native.events.length) {
                // Unchecking with nothing curated in the native panel yet:
                // seed it from whatever keyframe-space answer is already
                // set, so the two modes hand off smoothly instead of
                // starting from scratch. Skipped if the native panel
                // already has something (e.g. re-toggled back and forth)
                // -- never clobber curated work with a stale keyframe seed.
                const seeds = s.confirmed ? (s.answerFrame ? [s.answerFrame] : []) : s.answers;
                for (const f of seeds) {
                    try {
                        const info = await getExportFrame(f.video_id, f.n);
                        addEventToCuration("native", { video_id: f.video_id, frame_idx: info.frame_idx, thumbnail: info.thumbnail_url });
                    } catch (err) { /* unresolvable -- skip, not fatal */ }
                }
                if (seeds.length) await loadCurationVideo(seeds[0].video_id, "native", { seekN: seeds[0].n });
            }
        } finally {
            e.target.disabled = false;
        }
    };
    el("#exp-nbr-more").onclick = () => { s.neighboursShown += previewPage(); renderPreview(); };
    el("#exp-sim-more").onclick = () => { s.similarsShown += previewPage(); renderPreview(); };
    el("#exp-cancel").onclick = () => onDone("cancel");

    // Typed "Video ID" / "Frame ID" boxes + Change/Add event button --
    // KIS/VQA native mode (Keyframes unchecked) has no such row at all any
    // more (item 1.1: dropped entirely, VQA gets the answer box in its
    // place instead -- see renderTypeVisibility), so this only ever runs
    // for TRAKE or keyframe-mode KIS/VQA.
    // KIS/VQA keyframe mode: same destination as the preview-pick
    // (applyChangedFrame), but reaches an arbitrary frame not necessarily
    // in either preview list -- verifies the frame actually exists (via
    // /api/export/frame, n-based) before applying, so a typo lands as one
    // clear error rather than a broken export.
    // TRAKE: "Frame ID" is a raw native frame number, not a keyframe n,
    // added to whatever video the TRAKE curation panel below is already
    // on -- the (read-only) Video ID box is just a reminder of that, not
    // a second way to pick the video (see the panel's own Load/switch row
    // for that). No backend round-trip needed at all.
    el("#exp-change-btn").onclick = async () => {
        clearStatus();

        if (s.queryType === "TRAKE") {
            if (!s.trake.videoId) {
                showStatus("Load a video in the panel below first.");
                return;
            }
            const num = parseFrameIdInput(el("#exp-change-frame").value);
            if (!num) {
                showStatus("Enter a real frame number.");
                return;
            }
            if (addEventToCuration("trake", { video_id: s.trake.videoId, frame_idx: num })) {
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
            const exportBtn = el("#exp-export");
            // Nothing generated at all yet (fresh cache): rather than make
            // the user click "Generate rows" first, generate the currently
            // curated video's rows on the fly -- only for this true "0
            // rows anywhere" case, not e.g. "generated but unchecked",
            // which stays today's explicit error below.
            if (!s.trake.cache.size) {
                if (!s.trake.videoId || !s.trake.events.length) {
                    showStatus("Nothing to export -- curate a video, click \"Generate rows\", then check it below.");
                    return;
                }
                exportBtn.disabled = true;
                try {
                    await generateRowsForCurationVideo();
                } finally {
                    exportBtn.disabled = false;
                }
                if (!s.trake.cache.size) return; // generation failed -- generateRowsForCurationVideo() already showed why
            }

            // Client-side merge of the per-video cache -- no candidates/
            // confirmed/answers body to build, unlike KIS/VQA below.
            const merged = mergeTrakeCache(99);
            if (!merged.length) {
                showStatus("Nothing to export -- curate a video, click \"Generate rows\", then check it below.");
                return;
            }
            const filename = queryFilename("TRAKE", s.name);
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

        // KIS/VQA, Keyframes unchecked: the answer is whatever's curated in
        // the native panel (native frame_idx space) rather than
        // s.answerFrame/s.answers -- candidates/neighbours-by-time still
        // get computed server-side (item 2: same backend logic, no
        // preview), just from generate_export()'s keyframes=false branch
        // instead. Confirmed mode's "similar" tier there is a fresh visual
        // search snapped to the curated frame's nearest keyframe
        // (similar_candidates_for_native_frame) -- computed on the
        // backend regardless of what `candidates` carries here, same as
        // keyframe-mode confirmed below.
        if (!s.keyframes) {
            if (!s.native.events.length) {
                showStatus(s.confirmed
                    ? "No chosen frame -- play the video and click \"Switch to this frame\"."
                    : "Add at least one answer frame from the video.");
                return;
            }
            const body = {
                query_type: s.queryType,
                mode: s.confirmed ? "confirmed" : "unconfirmed",
                keyframes: false,
                candidates: s.confirmed ? [] : getCandidates(),
                confirmed: s.confirmed ? { video_id: s.native.videoId, frame_idx: s.native.events[0].frame_idx } : null,
                answers: s.confirmed ? [] : s.native.events.map((e) => ({ video_id: s.native.videoId, frame_idx: e.frame_idx })),
                answer: s.answerText,
                neighbour_count: NEIGHBOUR_COUNT_EXPORT,
                filename: queryFilename(s.queryType, s.name),
            };
            const exportBtn = el("#exp-export");
            exportBtn.disabled = true;
            try {
                await exportCsv(body);
                showStatus(`✓ Exported ${body.filename}.csv -- you can export again from here if needed.`, "info");
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
            keyframes: true,
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
    // all three query types are usable regardless (item 7): TRAKE and
    // native (Keyframes-unchecked) KIS/VQA work directly off the raw
    // frame_idx it carries; checking Keyframes back on for KIS/VQA snaps
    // to the nearest indexed keyframe (the #exp-keyframes handler above).

    renderTypeVisibility();
    renderAnswerContent();
    renderPreview();
    renderTrakeContent();
    renderNativeContent();

    // Seed the curation panel(s) straight from `trigger` when it carries a
    // video/frame of its own -- any signal's result card, a real TRAKE
    // candidate's own matched events, or a raw playback frame can all
    // start (or extend) a TRAKE sequence, not just a real TRAKE search
    // (see module docstring). Not limited to when queryType actually
    // starts on TRAKE -- switching to TRAKE later still finds the panel
    // already seeded, and likewise for KIS/VQA's native panel (a "frame"
    // trigger seeds both up front, below, since it has no way to know
    // which one the user will end up on).
    // loadCurationVideo is called right after kicking the event-seeding
    // off (not awaited first) so the video starts loading immediately
    // rather than waiting on the frame-info round trip(s) below; its own
    // videoId!==state.videoId check still resets state.events first, but
    // that's a no-op here since freshState() always starts empty.
    let seedVideoId = null;
    if (trigger.kind === "trake") {
        // Resolved in parallel but applied in original event order (matters
        // here, unlike a lone addEventFromN call elsewhere) -- several
        // concurrent fetches racing straight into addEventToCuration could
        // otherwise land E2 before E1 depending on which response arrives
        // first.
        seedVideoId = trigger.candidate.video_id;
        const matched = trigger.candidate.events.filter((e) => e.matched);
        Promise.all(matched.map((e) => getExportFrame(seedVideoId, e.n).catch(() => null))).then((infos) => {
            for (const info of infos) {
                if (info) addEventToCuration("trake", { video_id: seedVideoId, frame_idx: info.frame_idx, thumbnail: info.thumbnail_url });
            }
        });
    } else if (trigger.kind === "flat") {
        seedVideoId = trigger.video_id;
        addEventFromN("trake", trigger.video_id, trigger.n);
    } else if (trigger.kind === "frame") {
        seedVideoId = trigger.video_id;
        // thumbnail/current_time come from the playback dialog's own
        // canvas-snapshot + currentTime at the moment "Export this frame"
        // was clicked (dialogs.js) -- gives this event a real preview
        // instead of "no preview", and lets the curation video below
        // resume from the same spot instead of restarting at 0:00. Seeds
        // both the TRAKE and native KIS/VQA panels with the same frame --
        // TRAKE starts active (queryType default above), but Keyframes
        // defaults unchecked for this trigger too (freshState), so
        // switching straight to KIS/VQA already has this frame ready.
        addEventToCuration("trake", { video_id: trigger.video_id, frame_idx: trigger.frame_idx, thumbnail: trigger.thumbnail ?? null });
        addEventToCuration("native", { video_id: trigger.video_id, frame_idx: trigger.frame_idx, thumbnail: trigger.thumbnail ?? null });
    }
    if (seedVideoId) {
        loadCurationVideo(seedVideoId, "trake",
            trigger.kind === "frame" ? { seekTime: trigger.current_time || 0 }
                : trigger.kind === "flat" ? { seekN: trigger.n }
                    : {});
    }
}
