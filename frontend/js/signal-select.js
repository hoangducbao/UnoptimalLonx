// frontend/js/signal-select.js -- tiny shared <select> builder for a
// per-row signal dropdown, used by both TRAKE's event rows and Mixed's
// sub-query rows (each passes its own option list).

export function signalSelectHtml(id, current, options) {
    const opts = options.map((s) => `<option value="${s}"${s === current ? " selected" : ""}>${s}</option>`).join("");
    return `<select id="${id}" style="width:100%;padding:0.35rem;border:1px solid var(--border);border-radius:6px;">${opts}</select>`;
}
