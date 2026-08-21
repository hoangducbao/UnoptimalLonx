// frontend/js/signals/caption.js -- Caption signal panel. ui/app.py:1959-1961 (controls), 2026-2050 (render).

import { searchCaption } from "../api.js";
import { makeTextSignalPanel } from "./_text_signal.js";

const groupMode = () => document.getElementById("group-by-video").checked ? "video" : null;

export const { mount, run } = makeTextSignalPanel({
    prefix: "cap",
    siglipLabel: "SigLIP2 Caption",
    fuzzyLabel: "Fuzzy Caption",
    rrfLabel: "RRF Caption",
    searchFn: searchCaption,
    groupMode,
});
