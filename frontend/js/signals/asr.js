// frontend/js/signals/asr.js -- ASR signal panel. ui/app.py:1955-1957 (controls), 2000-2024 (render).
// Defaults/order deviate from ui/app.py here per explicit request: Fuzzy
// ASR listed first and on by default, SigLIP2/RRF off by default (the
// original defaulted all three on).

import { searchAsr } from "../api.js";
import { makeTextSignalPanel } from "./_text_signal.js";
import { settings } from "../settings.js";

const groupMode = () => settings.groupByVideo ? "video" : null;

export const { mount, run } = makeTextSignalPanel({
    prefix: "asr",
    siglipLabel: "SigLIP2 ASR",
    fuzzyLabel: "Fuzzy ASR",
    rrfLabel: "RRF ASR",
    searchFn: searchAsr,
    groupMode,
    order: ["fuzzy", "siglip", "rrf"],
    defaults: { siglip: false, fuzzy: true, rrf: false },
});
