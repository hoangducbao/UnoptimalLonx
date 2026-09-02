// frontend/js/signals/summary.js -- Summary signal panel. Video-level, not
// frame-level (ui/app.py:760-765): one result per video, so its "group by"
// groups by collection (lot) instead of by video (ui/app.py:1941-1944).

import { searchSummary } from "../api.js";
import { makeTextSignalPanel } from "./_text_signal.js";
import { settings, setGroupByUi } from "../settings.js";

const groupMode = () => settings.groupByVideo ? "collection" : null;

export const { mount: baseMount, run } = makeTextSignalPanel({
    prefix: "sum",
    siglipLabel: "SigLIP2 Summary",
    fuzzyLabel: "Fuzzy Summary",
    rrfLabel: "RRF Summary",
    searchFn: searchSummary,
    groupMode,
});

// Same group-by toggle everywhere (now in the Settings dialog), but its
// label/meaning flips to "Group by collection" for Summary specifically --
// ui/app.py does this by relabeling the same toggle, not adding a second
// one, so we mirror that here on mount/unmount instead of introducing a
// Summary-only setting.
const SUMMARY_LABEL = "Group by collection";

export function mount(controlsEl) {
    setGroupByUi({ label: SUMMARY_LABEL });
    baseMount(controlsEl);
}

export function unmount() {
    setGroupByUi();
}
