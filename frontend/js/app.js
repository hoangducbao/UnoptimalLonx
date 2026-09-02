// frontend/js/app.js -- wires the signal switcher, initial load, and
// query-submit trigger. Ports ui/app.py's segmented_control mode switch
// (ui/app.py:1611-1629) and the top-level `if mode == "...":` render
// dispatch. Only Keyframe is wired in Phase 1 -- later phases register
// more entries in SIGNALS and enable their sidebar buttons.

import { getProfile } from "./api.js";
import { resetExportCandidates, state } from "./state.js";
import { setOnSubmit } from "./query-input.js";
import { initFacets } from "./facets.js";
import { openSettingsDialog } from "./dialogs.js";
import { tile } from "./settings.js";
import * as keyframe from "./signals/keyframe.js";
import * as asr from "./signals/asr.js";
import * as caption from "./signals/caption.js";
import * as ocr from "./signals/ocr.js";
import * as summary from "./signals/summary.js";
import * as mixed from "./signals/mixed.js";
import * as trake from "./signals/trake.js";
import * as hierarchy from "./signals/hierarchy.js";

const SIGNALS = {
    Keyframe: keyframe,
    ASR: asr,
    Caption: caption,
    OCR: ocr,
    Summary: summary,
    Mixed: mixed,
    TRAKE: trake,
    Hierarchy: hierarchy,
};

const resultsEl = document.getElementById("results");
const statusEl = document.getElementById("status-banner");
const controlsEl = document.getElementById("signal-controls");

function currentModule() {
    return SIGNALS[state.signal];
}

function runCurrentSearch() {
    const mod = currentModule();
    if (mod) mod.run(resultsEl, statusEl);
}

function selectSignal(name) {
    if (!SIGNALS[name]) return; // not wired up yet (later phase)
    const prevMod = currentModule();
    if (prevMod?.unmount) prevMod.unmount();
    state.signal = name;
    document.querySelectorAll(".signal-btn[data-signal]").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.signal === name);
    });
    SIGNALS[name].mount(controlsEl);
    // A new signal's results are unrelated to the old one's -- clear the
    // export popup's "similars" source immediately rather than waiting for
    // the next search to land (some run()s return early without searching,
    // e.g. an empty query box).
    resetExportCandidates([]);
    runCurrentSearch();
}

document.querySelectorAll(".signal-btn[data-signal]").forEach((btn) => {
    btn.addEventListener("click", () => selectSignal(btn.dataset.signal));
});

// ⚙ sits in the same icon row but isn't a signal -- it opens the settings
// dialog (hover zoom, tile size, result counts, grouping) instead of
// switching modes. Everything in there can change what a search returns or
// how it's grouped, so a Save just re-runs the current one; the dialog
// itself writes any changed Top-K/V/G back into the sidebar boxes.
document.getElementById("top-g").value = tile().topG; // Top-G's default is a property of the tile size
document.getElementById("settings-btn").addEventListener("click", () => {
    openSettingsDialog(runCurrentSearch);
});

// Re-run search on any sidebar control change (mirrors Streamlit's
// rerun-on-any-widget-interaction model, but only for the controls that
// actually affect a search -- clicking "Show more"/"Copy" doesn't touch
// these listeners at all, so there's no wasted-recompute problem to guard
// against here in the first place).
// ("Group by video"/"Show full text" aren't here any more -- they're saved
// settings now, and the ⚙ dialog's Save re-runs the search itself.)
["top-k", "top-v", "top-g", "video-filter", "lot-filter",
 "facet-value"].forEach((id) => {
    document.getElementById(id).addEventListener("change", runCurrentSearch);
});
// facet-field is wired by initFacets() below instead of the generic list
// above: switching field must reset facet-value's options *before*
// runCurrentSearch reads it, and that ordering can't be guaranteed across
// two independently-registered "change" listeners.

// Scope segmented control ("vid" / "coll" / "excl") -- at most one active
// at a time (0 or 1, never 2-3): clicking an inactive tile clears its
// siblings before activating it; clicking the already-active one just
// turns it off, back to 0. Read back by state.js::scopeFilters() via each
// button's .active class, not a checkbox's .checked.
document.querySelectorAll("#scope-segmented button").forEach((btn) => {
    btn.addEventListener("click", () => {
        const wasActive = btn.classList.contains("active");
        document.querySelectorAll("#scope-segmented button").forEach((b) => b.classList.remove("active"));
        if (!wasActive) btn.classList.add("active");
        runCurrentSearch();
    });
});

document.getElementById("video-filter").addEventListener("input", () => {}); // no live-search on keystroke; Enter/blur via change above
document.getElementById("clear-image-query").addEventListener("click", runCurrentSearch);

setOnSubmit(runCurrentSearch);

// Delegate signal-control checkbox changes (leg toggles etc., re-created
// per signal by mount()) up through the container.
controlsEl.addEventListener("change", runCurrentSearch);

selectSignal("Keyframe");
initFacets(runCurrentSearch);

// Which embedding profile the backend behind THIS tab loaded (768 vs 1152 --
// see backend/config.py). The two run as separate processes on separate
// ports and are otherwise pixel-identical, so without this badge it's only a
// matter of time before a result gets credited to the wrong model. Also
// stamped into the tab title, for when the tab is too narrow to read.
getProfile().then(({ profile, dim, model_id }) => {
    const el = document.getElementById("profile-badge");
    el.textContent = `${profile}d`;
    el.dataset.profile = profile;
    el.title = `${dim}-dim embeddings — ${model_id}`;
    document.title = `Routing101 (${profile}d)`;
}).catch(() => { /* badge is informational; a failed fetch shouldn't break the app */ });
