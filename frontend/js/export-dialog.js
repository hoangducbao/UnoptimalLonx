// frontend/js/export-dialog.js -- opens the Export CSV UI (export-ui.js,
// mounted by export-page.js) in a new browser tab (frontend/export.html)
// instead of an in-page modal, so the export workflow survives switching
// search tabs/queries/signals in the original window and can sit side by
// side with the results that fed it.
//
// Hands the trigger across the tab boundary via a same-origin
// window.opener handoff (state.js exposes window.__routing101 for this) --
// no serialization needed: as long as the original tab stays open, both
// windows can read/write the same JS objects directly, including a LIVE
// reference to the opener's exportState (see export-page.js), so
// "Similars" in the new tab reflects the opener's most recent search, not
// a frozen snapshot from whenever the tab was opened.
//
// `trigger` shapes (unchanged, see export-ui.js):
//   {kind: "flat", video_id, n}   -- any non-TRAKE signal's result card
//   {kind: "trake", candidate}    -- a TRAKE candidate card

import "./state.js"; // side-effect only -- ensures window.__routing101 exists before the click below

export function openExportDialog(trigger) {
    const id = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`;
    window.__routing101.handoffs.set(id, trigger);
    window.open(`/app/export.html?handoff=${id}`, "_blank");
}
