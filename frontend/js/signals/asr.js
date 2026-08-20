// frontend/js/signals/asr.js -- ASR signal panel. ui/app.py:1955-1957 (controls), 2000-2024 (render).

import { searchAsr } from "../api.js";
import { makeTextSignalPanel } from "./_text_signal.js";

const groupMode = () => document.getElementById("group-by-video").checked ? "video" : null;

export const { mount, run } = makeTextSignalPanel({
    prefix: "asr",
    siglipLabel: "SigLIP2 ASR",
    fuzzyLabel: "Fuzzy ASR",
    rrfLabel: "RRF ASR",
    searchFn: searchAsr,
    groupMode,
});
