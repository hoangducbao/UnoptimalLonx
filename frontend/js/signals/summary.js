// frontend/js/signals/summary.js -- Summary signal panel. Video-level, not
// frame-level (ui/app.py:760-765): one result per video, so its "group by"
// groups by collection (lot) instead of by video (ui/app.py:1941-1944).

import { searchSummary } from "../api.js";
import { makeTextSignalPanel } from "./_text_signal.js";

const groupMode = () => document.getElementById("group-by-video").checked ? "collection" : null;

export const { mount: baseMount, run } = makeTextSignalPanel({
    prefix: "sum",
    siglipLabel: "SigLIP2 Summary",
    fuzzyLabel: "Fuzzy Summary",
    rrfLabel: "RRF Summary",
    searchFn: searchSummary,
    groupMode,
});

// Same "Group by video" checkbox everywhere, but its label/meaning flips to
// "Group by collection" for Summary specifically -- ui/app.py does this by
// relabeling the same toggle, not adding a second one, so we mirror that
// here on mount/unmount instead of introducing a Summary-only checkbox.
const groupLabel = document.querySelector('label[for="group-by-video"]');
const DEFAULT_LABEL = "Group by video";
const SUMMARY_LABEL = "Group by collection";

export function mount(controlsEl) {
    groupLabel.textContent = SUMMARY_LABEL;
    baseMount(controlsEl);
}

export function unmount() {
    groupLabel.textContent = DEFAULT_LABEL;
}
