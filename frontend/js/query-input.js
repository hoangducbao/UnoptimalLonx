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

// Query box sizing. Two things drive the height and they compose rather than
// fight: the box auto-grows to fit whatever is typed (uncapped -- a long
// query pushes the rest of the sidebar down and the sidebar scrolls), and
// #query-resize, the full-width bar under it standing in for the native
// corner grip (see css/style.css), sets a floor the box will not shrink
// below. So a drag makes the box taller than its text and keeps it there;
// typing past that floor keeps growing from it; deleting text shrinks back
// down to the floor, never past it.
const resizeBar = document.getElementById("query-resize");
const MIN_QUERY_HEIGHT = 64;   // matches .sidebar textarea's min-height

// Set by a drag, and the one piece of state here: null means "no floor of
// your own, just fit the text (down to MIN_QUERY_HEIGHT)".
let draggedHeight = null;

function fitToContent() {
    // Measure from scratch -- scrollHeight only ever reports the content's
    // full extent when the box isn't already sized to hold it, so shrinking
    // needs the height cleared first. The delta is the borders: box-sizing
    // is border-box here, and scrollHeight excludes them.
    textarea.style.height = "auto";
    const borders = textarea.offsetHeight - textarea.clientHeight;
    const content = textarea.scrollHeight + borders;
    textarea.style.height = `${Math.max(MIN_QUERY_HEIGHT, draggedHeight ?? 0, content)}px`;
}

// "input" covers typing, pasted text, cut, undo and drag-and-drop of text
// alike -- every path that can change the value from the user's side.
textarea.addEventListener("input", fitToContent);
fitToContent();   // a value restored by the browser on reload starts fitted

// Pointer events rather than mouse ones so a touch/pen drag works the same;
// capture keeps the drag alive when the pointer leaves the 10px bar, which
// it does at once.
let dragStartY = 0;
let dragStartHeight = 0;

resizeBar.addEventListener("pointerdown", (e) => {
    e.preventDefault();   // no text selection while dragging
    dragStartY = e.clientY;
    dragStartHeight = textarea.getBoundingClientRect().height;
    resizeBar.setPointerCapture(e.pointerId);
    resizeBar.classList.add("dragging");
});

resizeBar.addEventListener("pointermove", (e) => {
    if (!resizeBar.hasPointerCapture(e.pointerId)) return;
    draggedHeight = Math.max(MIN_QUERY_HEIGHT, dragStartHeight + (e.clientY - dragStartY));
    fitToContent();   // the floor moved; content may still want more
});

for (const ev of ["pointerup", "pointercancel"]) {
    resizeBar.addEventListener(ev, (e) => {
        resizeBar.releasePointerCapture(e.pointerId);
        resizeBar.classList.remove("dragging");
    });
}

// Double-click the bar to drop the dragged floor and go back to hugging the
// text -- the one thing a free drag makes awkward to get back to.
resizeBar.addEventListener("dblclick", () => {
    draggedHeight = null;
    fitToContent();
});

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
