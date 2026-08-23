// frontend/js/export-dialog.js -- the export popup: opened by the star
// button on any result card (render.js/trake.js). Two bodies depending on
// query type:
//   KIS/VQA -- one frame's worth of query-answer curation plus a two-
//     section preview (nearest-by-time "Neighbours", already-ranked
//     "Similars").
//   TRAKE -- confirmed mode: one frame_idx input per event, seeded from
//     the candidate's own search-matched frame and refined by ear/eye via
//     the existing marker-bar playback dialog (native frame_idx, not tied
//     to any keyframe -- see backend/export.py's module docstring);
//     unconfirmed mode: a read-only summary, since the whole current
//     TRAKE candidate list exports as-is (no per-frame curation step).
// Either way, POSTs to /api/export and triggers the CSV download.
//
// `trigger` shapes:
//   {kind: "flat", video_id, n}   -- any non-TRAKE signal's result card
//   {kind: "trake", candidate}    -- a TRAKE candidate card

import { exportCsv, getExportFrame, getExportNeighbors } from "./api.js";
import { openDialog, openTrakePlaybackDialog } from "./dialogs.js";
import { exportState } from "./state.js";

const NEIGHBOUR_COUNT_EXPORT = 10; // fixed row-generation window, independent of preview expand state
const PREVIEW_PAGE = 12; // 3x4 grid per preview section (export-preview-grid is 4 columns wide)

function freshState(trigger) {
    const isFlat = trigger.kind === "flat";
    const seed = isFlat ? { video_id: trigger.video_id, n: trigger.n } : null;
    const trakeEvents = trigger.kind === "trake" ? trigger.candidate.events : [];
    return {
        trigger,
        queryType: isFlat ? "KIS" : "TRAKE",
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
        // TRAKE confirmed mode: video_id + one editable native frame_idx
        // per event, seeded (once fetched) from that event's own search
        // match -- see prefillTrakeFrames().
        trakeVideoId: trigger.kind === "trake" ? trigger.candidate.video_id : null,
        trakeEvents,
        trakeFrameIdxs: trakeEvents.map(() => null),
    };
}

function frameKey(f) { return `${f.video_id}|${f.n}`; }

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

export function openExportDialog(trigger) {
    const s = freshState(trigger);

    const body = document.createElement("div");
    body.className = "export-dialog";
    body.innerHTML = `
        <div class="export-options">
          <div class="segmented" id="exp-segmented">
            <button type="button" data-type="KIS">KIS</button>
            <button type="button" data-type="VQA">VQA</button>
            <button type="button" data-type="TRAKE">TRAKE</button>
          </div>
          <input type="number" id="exp-name" min="1" step="1" placeholder="Query # -- exported as query-p2-<#>-<type>.csv">
        </div>
        <div class="checkbox-row">
          <input type="checkbox" id="exp-confirmed" checked>
          <label for="exp-confirmed" style="margin:0;">Confirmed</label>
        </div>
        <div id="exp-flat-body">
          <div class="export-answer-area">
            <div id="exp-answer-content"></div>
            <input type="text" id="exp-answer-text" placeholder="VQA answer" style="display:none;">
          </div>
          <div class="export-preview-area">
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
          </div>
        </div>
        <div id="exp-trake-body" style="display:none;">
          <div id="exp-trake-content"></div>
        </div>
        <div class="status-banner error" id="exp-error" style="display:none;"></div>
        <div class="export-actions">
          <button class="btn" id="exp-cancel">Cancel</button>
          <button class="btn btn-primary" id="exp-export">⬇ Export</button>
        </div>`;
    const { overlay, box } = openDialog("Export CSV", body, { wide: true });

    const el = (sel) => box.querySelector(sel);

    function renderTypeVisibility() {
        el("#exp-segmented").querySelectorAll("button").forEach((btn) => {
            btn.classList.toggle("active", btn.dataset.type === s.queryType);
        });
        const isTrake = s.queryType === "TRAKE";
        el("#exp-trake-body").style.display = isTrake ? "block" : "none";
        el("#exp-flat-body").style.display = isTrake ? "none" : "block";

        const isVqa = s.queryType === "VQA";
        const answerText = el("#exp-answer-text");
        answerText.style.display = isVqa ? "block" : "none";
        answerText.disabled = isVqa && !s.confirmed;
        answerText.placeholder = (isVqa && !s.confirmed) ? "LLM needed (filled automatically -- later phase)" : "VQA answer";
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
                content.innerHTML = `<div class="status-banner info">No frame selected -- open this dialog from a result card's ★ button.</div>`;
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
        renderAnswerContent();
        renderPreview();
    }

    function previewCardHtml(f, { addable }) {
        const already = addable && isInAnswers(f);
        const addBtn = addable
            ? `<button class="icon-btn export-add-btn${already ? " added" : ""}" title="${already ? "Already added" : "Add to answer(s)"}" data-video-id="${f.video_id}" data-n="${f.n}">${already ? "✓" : "+"}</button>`
            : "";
        return `<div class="thumb-cell">
            <div class="thumb-wrap"><img src="${f.thumbnail_url}" loading="lazy"></div>
            <div class="thumb-caption"><b>${f.video_id}</b> · frame ${f.n}</div>
            ${addBtn}
        </div>`;
    }

    async function renderPreview() {
        const addable = !s.confirmed;

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
            nbrGrid.innerHTML = frames.map((f) => previewCardHtml(f, { addable })).join("") || `<div class="status-banner info">No neighbours found.</div>`;
            el("#exp-nbr-more").style.display = s.neighbourFrames.length >= s.neighboursShown ? "block" : "none";
        }

        // Similars -- the query's own already-fetched, already-ranked results.
        const simGrid = el("#exp-sim-grid");
        const similars = exportState.candidates.slice(0, s.similarsShown);
        simGrid.innerHTML = similars.map((c) => previewCardHtml(c, { addable }))
            .join("") || `<div class="status-banner info">No results from the last search.</div>`;
        el("#exp-sim-more").style.display = exportState.candidates.length > s.similarsShown ? "block" : "none";

        if (addable) {
            box.querySelectorAll(".export-add-btn").forEach((btn) => {
                btn.onclick = () => addToAnswers({ video_id: btn.dataset.videoId, n: Number(btn.dataset.n) });
            });
        }
    }

    // TRAKE confirmed mode: seed each event's editable frame_idx from its
    // own search match (n -> frame_idx), once, on open. Unmatched events
    // are left blank -- there's nothing to seed from, the human has to
    // watch and fill them in from scratch. Never overwrites a value the
    // human has already typed (checked at resolve time, not call time, in
    // case two fetches for different events land close together).
    function prefillTrakeFrames() {
        s.trakeEvents.forEach((e, i) => {
            if (!e.matched) return;
            getExportFrame(e.video_id, e.n).then((info) => {
                if (s.trakeFrameIdxs[i] === null) {
                    s.trakeFrameIdxs[i] = info.frame_idx;
                    renderTrakeContent();
                }
            }).catch(() => {});
        });
    }

    function renderTrakeContent() {
        const content = el("#exp-trake-content");
        if (s.trigger.kind !== "trake") {
            content.innerHTML = `<div class="status-banner info">Open this dialog from a TRAKE candidate's ★ button.</div>`;
            return;
        }

        if (s.confirmed) {
            const rows = s.trakeEvents.map((e, i) => {
                const val = s.trakeFrameIdxs[i];
                const thumb = e.matched
                    ? `<div class="thumb-wrap thumb-wrap-static"><img src="${e.thumbnail_url}" loading="lazy"></div>`
                    : `<div class="thumb-missing">no match</div>`;
                return `<div class="trake-event-row">
                    ${thumb}
                    <div class="trake-event-fields">
                        <div class="thumb-caption"><b>${e.label}</b>${e.matched ? "" : " · not found by search"}</div>
                        <input type="number" class="trake-frame-input" data-index="${i}" min="0" step="1"
                               placeholder="frame_idx" value="${val === null ? "" : val}">
                    </div>
                </div>`;
            }).join("");
            content.innerHTML = `
                <div class="thumb-caption" style="margin-bottom:0.5rem;">
                    <b>${s.trakeVideoId}</b>
                    <button class="btn" id="trake-play-btn" type="button">▶ Play video</button>
                </div>
                <div class="thumb-caption muted" style="margin-bottom:0.5rem;">
                    Each field starts at that event's search match (its own keyframe's native frame) --
                    play the video, read the frame counter, and correct any that are off.
                </div>
                <div class="trake-event-list">${rows}</div>`;
            el("#trake-play-btn").onclick = () => openTrakePlaybackDialog(s.trakeVideoId, s.trakeEvents);
            content.querySelectorAll(".trake-frame-input").forEach((input) => {
                input.oninput = () => {
                    const i = Number(input.dataset.index);
                    s.trakeFrameIdxs[i] = input.value === "" ? null : Number(input.value);
                };
            });
        } else {
            const candidates = exportState.candidates || [];
            if (!candidates.length) {
                content.innerHTML = `<div class="status-banner info">No TRAKE candidates from the last search.</div>`;
                return;
            }
            const rows = candidates.map((c) => {
                const matched = c.events.filter((e) => e.matched).length;
                return `<div class="trake-candidate-row"><b>${c.video_id}</b> · video_score=${c.video_score.toFixed(4)} · coverage ${matched}/${c.events.length}</div>`;
            }).join("");
            content.innerHTML = `
                <div class="thumb-caption muted" style="margin-bottom:0.5rem;">
                    Exports all ${candidates.length} candidates from the current TRAKE search as-is, ranked by
                    confidence -- one row per video, then hedge rows fill the rest. No per-frame curation in
                    unconfirmed mode.
                </div>${rows}`;
        }
    }

    // --- wiring ---------------------------------------------------------

    el("#exp-segmented").querySelectorAll("button").forEach((btn) => {
        btn.onclick = () => { s.queryType = btn.dataset.type; renderTypeVisibility(); renderTrakeContent(); };
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
    el("#exp-cancel").onclick = () => overlay.remove();

    el("#exp-export").onclick = async () => {
        const errEl = el("#exp-error");
        errEl.style.display = "none";

        if (!s.name) {
            errEl.textContent = "Enter a query number.";
            errEl.style.display = "block";
            return;
        }

        let body;
        if (s.queryType === "TRAKE") {
            if (s.trigger.kind !== "trake") {
                errEl.textContent = "No TRAKE candidate -- open this dialog from a TRAKE result card.";
                errEl.style.display = "block";
                return;
            }
            if (s.confirmed) {
                if (s.trakeFrameIdxs.some((f) => f === null || f < 0)) {
                    errEl.textContent = "Fill in every event's frame_idx (watch the video with ▶ to find any missing ones).";
                    errEl.style.display = "block";
                    return;
                }
            } else if (!exportState.candidates.length) {
                errEl.textContent = "No TRAKE candidates from the last search.";
                errEl.style.display = "block";
                return;
            }
            body = {
                query_type: "TRAKE",
                mode: s.confirmed ? "confirmed" : "unconfirmed",
                candidates: exportState.candidates,
                confirmed: s.confirmed ? { video_id: s.trakeVideoId, frame_idxs: s.trakeFrameIdxs } : null,
                answers: [],
                answer: "",
                filename: queryFilename("TRAKE", s.name),
            };
        } else {
            if (s.confirmed && !s.answerFrame) {
                errEl.textContent = "No confirmed frame -- open this dialog from a result card.";
                errEl.style.display = "block";
                return;
            }
            if (!s.confirmed && !s.answers.length) {
                errEl.textContent = "Add at least one answer frame from the preview.";
                errEl.style.display = "block";
                return;
            }
            body = {
                query_type: s.queryType,
                mode: s.confirmed ? "confirmed" : "unconfirmed",
                candidates: exportState.candidates,
                confirmed: s.confirmed ? s.answerFrame : null,
                answers: s.confirmed ? [] : s.answers,
                answer: s.answerText,
                neighbour_count: NEIGHBOUR_COUNT_EXPORT,
                filename: queryFilename(s.queryType, s.name),
            };
        }

        const exportBtn = el("#exp-export");
        exportBtn.disabled = true;
        try {
            await exportCsv(body);
            overlay.remove();
        } catch (e) {
            errEl.textContent = e.message;
            errEl.style.display = "block";
        } finally {
            exportBtn.disabled = false;
        }
    };

    renderTypeVisibility();
    renderAnswerContent();
    renderPreview();
    renderTrakeContent();
    if (trigger.kind === "trake") prefillTrakeFrames();
}
