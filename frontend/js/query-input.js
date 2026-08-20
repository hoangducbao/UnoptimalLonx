// frontend/js/query-input.js -- query textarea (Enter-to-submit) + paste-to-
// image handling. Ports the *behavior* of ui/app.py's two JS injections
// (ui/app.py:1664-1778) but trivially: a hand-written page has no
// "rerun the whole script" model to defend against, so this is just plain
// addEventListener calls at module load -- no MutationObserver, no
// singleton-guard dance (that machinery in ui/app.py existed purely to
// survive Streamlit re-emitting the same <script> block on every rerun).

import { uploadQueryImage } from "./api.js";
import { state } from "./state.js";

const textarea = document.getElementById("query-text");
const preview = document.getElementById("image-query-preview");
const previewImg = document.getElementById("image-query-thumb");
const clearBtn = document.getElementById("clear-image-query");

let onSubmit = () => {};
export function setOnSubmit(fn) { onSubmit = fn; }

textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.isComposing) {
        e.preventDefault();
        onSubmit();
    }
});

textarea.addEventListener("paste", async (e) => {
    const items = e.clipboardData?.items || [];
    for (const item of items) {
        if (item.type.startsWith("image")) {
            e.preventDefault();
            const blob = item.getAsFile();
            const { image_id } = await uploadQueryImage(blob);
            state.imageQueryId = image_id;
            previewImg.src = URL.createObjectURL(blob);
            preview.style.display = "flex";
            onSubmit();
            return;
        }
    }
});

clearBtn.addEventListener("click", () => {
    state.imageQueryId = null;
    preview.style.display = "none";
    previewImg.src = "";
});

export function currentQuery() {
    return { query: textarea.value, image_id: state.imageQueryId };
}
