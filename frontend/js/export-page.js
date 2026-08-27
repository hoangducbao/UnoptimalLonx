// frontend/js/export-page.js -- entry point for the standalone Export CSV
// tab (frontend/export.html), opened via window.open from
// export-dialog.js's openExportDialog(). Reads the handed-off trigger and a
// live reference to the opener's exportState from
// window.opener.__routing101 (see state.js), then mounts the same UI
// export-ui.js used everywhere a ★ button leads.

import { buildExportUI } from "./export-ui.js";

const root = document.getElementById("export-root");

function fail(message) {
    root.innerHTML = `<div class="status-banner error">${message}</div>`;
}

const REOPEN_HINT = "reopen it from a search result's ★ button in the original tab.";

const params = new URLSearchParams(location.search);
const handoffId = params.get("handoff");
const opener = window.opener;

if (!handoffId || !opener || opener.closed || !opener.__routing101) {
    fail(`This export tab has no source page to read from -- ${REOPEN_HINT}`);
} else {
    // One-shot: consumed here so a stale/duplicated handoff id (or a
    // reload of this tab, which re-runs this script from scratch) can't
    // silently reuse a trigger meant for a different tab.
    const trigger = opener.__routing101.handoffs.get(handoffId);
    opener.__routing101.handoffs.delete(handoffId);

    if (!trigger) {
        fail(`This export link already expired -- ${REOPEN_HINT}`);
    } else {
        buildExportUI(root, trigger, {
            // Guarded against the opener tab closing mid-session -- degrades
            // to an empty Similars list rather than throwing.
            getCandidates: () => {
                try {
                    return (!opener.closed && opener.__routing101.exportState.candidates) || [];
                } catch (e) {
                    return [];
                }
            },
            onDone: (reason, filename) => {
                if (reason === "exported") {
                    root.innerHTML = `<div class="status-banner info">✓ Exported ${filename}.csv -- you can close this tab.</div>`;
                    return;
                }
                // Cancel: this tab only exists because we opened it via
                // window.open(), so window.close() is allowed to work here
                // -- but some browsers still refuse it, hence the fallback.
                window.close();
                setTimeout(() => {
                    root.innerHTML = `<div class="status-banner info">Cancelled -- you can close this tab now.</div>`;
                }, 300);
            },
        });
    }
}
