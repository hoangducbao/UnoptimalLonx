// frontend/js/format.js -- tiny shared display-formatting helpers with no
// state/dependencies of their own, split out so dialogs.js and
// export-ui.js's playback timers don't each carry their own copy.

export function fmtTime(t) {
    const mm = String(Math.floor(t / 60)).padStart(2, "0");
    const ss = (t % 60).toFixed(2).padStart(5, "0");
    return `${mm}:${ss}`;
}
