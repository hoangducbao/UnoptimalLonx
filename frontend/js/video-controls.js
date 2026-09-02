// frontend/js/video-controls.js -- shared <video> element behavior used by
// every playback surface (openPlaybackDialog/openTrakePlaybackDialog in
// dialogs.js, and the TRAKE curation panel in export-ui.js): remembers the
// viewer's last-used volume/playback speed across sessions (localStorage,
// same persisted-config pattern as state.js's mixedConfig) and adds
// keyboard shortcuts for changing speed, since the native <video controls>
// UI exposes no speed control at all.

const STORAGE_KEY = "routing101_video_prefs";
const SPEED_STEP = 0.25;
const MIN_SPEED = 0.25;
const MAX_SPEED = 3;

// 0.25 (25%) rather than the browser's usual 1.0 default -- chosen so a
// freshly opened video never starts blasting at full volume.
const DEFAULT_VOLUME = 0.25;

function loadPrefs() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (raw) {
            const parsed = JSON.parse(raw);
            return {
                volume: typeof parsed.volume === "number" ? parsed.volume : DEFAULT_VOLUME,
                rate: typeof parsed.rate === "number" ? parsed.rate : 1,
            };
        }
    } catch (e) { /* corrupt/old value -- fall through to defaults */ }
    return { volume: DEFAULT_VOLUME, rate: 1 };
}

export const videoPrefs = loadPrefs();

function savePrefs() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(videoPrefs));
}

// Applies the remembered volume/speed to a freshly created <video>, then
// keeps them in sync with whatever the viewer changes next -- via the
// native controls for volume, via bindSpeedShortcut (or the native menu,
// on browsers that expose one) for speed -- so "last used" becomes the
// default for every video opened after this one, including after a reload.
export function applyVideoPrefs(video) {
    video.volume = videoPrefs.volume;
    video.playbackRate = videoPrefs.rate;
    video.addEventListener("volumechange", () => {
        if (video.muted) return; // don't clobber the remembered volume with 0 on mute
        videoPrefs.volume = video.volume;
        savePrefs();
    });
    video.addEventListener("ratechange", () => {
        videoPrefs.rate = video.playbackRate;
        savePrefs();
    });
}

// Keyboard speed control: "<"/"," slows down and ">"/"." speeds up by
// SPEED_STEP (matches either shift state of those keys, so plain ","/"."
// work too, not just Shift+ -- same keys YouTube uses for this), "0"
// resets to 1x. Bound on `document` (a dialog's own box isn't guaranteed
// to have focus, or contain the focus, just because it's open) but guarded
// by `target.isConnected` -- `target` is the dialog box (or the curation
// panel's video column), so once that's torn down/replaced the shortcut
// goes inert on its own even if a caller forgets to call the returned
// unbind. Also ignored while a text input/textarea anywhere has focus, so
// it doesn't fight typing elsewhere in the same dialog (e.g. the TRAKE
// curation panel's own text inputs). `speedLabel`, if given, is kept in
// sync with a live "1.25x" readout.
// Grabs a JPEG data URL of whatever frame `video` is showing right now --
// used anywhere a raw native frame needs a thumbnail but has no existing
// thumbnail file to point at the way a keyframe does (a TRAKE curation
// event, or a playback dialog's "Export this frame" handoff). The video
// element is expected to be same-origin (served from this app's own
// /media/video mount), so the canvas isn't tainted; still guarded in case
// a frame isn't decoded yet.
export function captureVideoThumbnail(video) {
    try {
        const w = video.videoWidth || 320, h = video.videoHeight || 180;
        const canvas = document.createElement("canvas");
        canvas.width = 160;
        canvas.height = Math.round(160 * (h / w));
        canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
        return canvas.toDataURL("image/jpeg", 0.7);
    } catch (e) {
        return null;
    }
}

export function bindSpeedShortcut(video, target, speedLabel) {
    function renderLabel() {
        if (speedLabel) speedLabel.textContent = `${video.playbackRate.toFixed(2).replace(/\.?0+$/, "")}x`;
    }
    function setSpeed(rate) {
        video.playbackRate = Math.max(MIN_SPEED, Math.min(MAX_SPEED, rate));
    }
    function onKeydown(e) {
        if (!target.isConnected) return;
        const tag = document.activeElement && document.activeElement.tagName;
        if (tag === "INPUT" || tag === "TEXTAREA") return;
        if (e.key === "<" || e.key === ",") { setSpeed(video.playbackRate - SPEED_STEP); e.preventDefault(); }
        else if (e.key === ">" || e.key === ".") { setSpeed(video.playbackRate + SPEED_STEP); e.preventDefault(); }
        else if (e.key === "0") { setSpeed(1); e.preventDefault(); }
    }
    document.addEventListener("keydown", onKeydown);
    video.addEventListener("ratechange", renderLabel);
    renderLabel();
    return () => document.removeEventListener("keydown", onKeydown);
}
