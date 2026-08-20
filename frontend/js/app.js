// frontend/js/app.js -- wires the signal switcher, initial load, and
// query-submit trigger. Ports ui/app.py's segmented_control mode switch
// (ui/app.py:1611-1629) and the top-level `if mode == "...":` render
// dispatch. Only Keyframe is wired in Phase 1 -- later phases register
// more entries in SIGNALS and enable their sidebar buttons.

import { state } from "./state.js";
import { setOnSubmit } from "./query-input.js";
import * as keyframe from "./signals/keyframe.js";
import * as asr from "./signals/asr.js";
import * as caption from "./signals/caption.js";
import * as ocr from "./signals/ocr.js";
import * as summary from "./signals/summary.js";

const SIGNALS = {
    Keyframe: keyframe,
    ASR: asr,
    Caption: caption,
    OCR: ocr,
    Summary: summary,
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
    document.querySelectorAll(".signal-btn").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.signal === name);
    });
    SIGNALS[name].mount(controlsEl);
    runCurrentSearch();
}

document.querySelectorAll(".signal-btn").forEach((btn) => {
    btn.addEventListener("click", () => selectSignal(btn.dataset.signal));
});

// Re-run search on any sidebar control change (mirrors Streamlit's
// rerun-on-any-widget-interaction model, but only for the controls that
// actually affect a search -- clicking "Show more"/"Copy" doesn't touch
// these listeners at all, so there's no wasted-recompute problem to guard
// against here in the first place).
["top-k", "video-filter", "use-video-scope", "lot-filter", "use-collection-scope",
 "group-by-video", "show-full-text"].forEach((id) => {
    document.getElementById(id).addEventListener("change", runCurrentSearch);
});

document.getElementById("video-filter").addEventListener("input", () => {}); // no live-search on keystroke; Enter/blur via change above
document.getElementById("clear-image-query").addEventListener("click", runCurrentSearch);

setOnSubmit(runCurrentSearch);

// Delegate signal-control checkbox changes (leg toggles etc., re-created
// per signal by mount()) up through the container.
controlsEl.addEventListener("change", runCurrentSearch);

selectSignal("Keyframe");
