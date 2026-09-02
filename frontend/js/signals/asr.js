// frontend/js/signals/asr.js -- ASR signal panel. ui/app.py:1955-1957 (controls), 2000-2024 (render).
// Defaults/order deviate from ui/app.py here per explicit request: the two
// text legs (Fuzzy, Exact) come first and are on by default, SigLIP2/RRF
// off (the original defaulted all three on and had no Exact leg at all).
// Exact is ASR-only -- Caption/Summary use the factory's default three.

import { searchAsr } from "../api.js";
import { makeTextSignalPanel } from "./_text_signal.js";
import { settings } from "../settings.js";

const groupMode = () => settings.groupByVideo ? "video" : null;

export const { mount, run } = makeTextSignalPanel({
    prefix: "asr",
    siglipLabel: "SigLIP2 ASR",
    fuzzyLabel: "Fuzzy ASR",
    exactLabel: "Exact ASR",
    rrfLabel: "RRF ASR",
    searchFn: searchAsr,
    groupMode,
    order: ["fuzzy", "exact", "siglip", "rrf"],
    defaults: { fuzzy: true, exact: true, siglip: false, rrf: false },
});
