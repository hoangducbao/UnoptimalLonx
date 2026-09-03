// frontend/js/settings.js -- display preferences (hover-zoom strength, tile
// display size) shared by the main search page and the standalone Export CSV
// tab. Persisted to localStorage, same as mixedConfig (state.js): one saved
// value read by whichever tab is open, rather than per-tab state.
//
// Tile size is a single knob driving every grid width *and* the counts that
// have to move with it: bigger tiles mean fewer per row, so a "show more"
// page that used to fill 3 rows still fills 3 rows. Every one of those
// numbers lives in TILE_SIZES below -- nothing else hardcodes a column count
// or a page size.

export const SETTINGS_DEFAULTS = {
    hoverZoom: 1.25,
    tileSize: "medium",
    // Moved here out of the sidebar -- these two used to be checkbox rows in
    // index.html read straight off the DOM; now they're saved preferences the
    // settings dialog owns, and every signal reads them from `settings`.
    groupByVideo: true,
    showFullText: false,
};

// Sidebar Top-K/Top-V defaults, mirrored by the settings dialog's own boxes
// (Top-G's default is per tile size, below). Not persisted -- these stay
// session state on the sidebar inputs, same as before; the dialog just offers
// a second way to set them and a way back to these numbers.
export const TOP_K_DEFAULT = 200;
export const TOP_V_DEFAULT = 15;

export const HOVER_ZOOM_MIN = 1.0;
export const HOVER_ZOOM_MAX = 3.0;
export const HOVER_ZOOM_STEP = 0.2;

export const TILE_SIZES = {
    // columns         -- main result grid + the nearby-frames popup grid
    // neighborsBefore -- nearby-frames popup: frames shown before the center
    // neighborsAfter  -- ...and after it (before + 1 center + after = 3 rows)
    // neighborStep    -- "▲ N earlier"/"▼ N later" step in that popup (2 rows)
    // previewColumns  -- Export tab's Neighbours/Similars preview grids
    // previewPage     -- those grids' initial count and "Show N more" step
    // topG            -- Hierarchy's sidebar Top-G default
    // hierExpand      -- Hierarchy's per-video "Expand" step
    small: {
        label: "Small",
        columns: 6, neighborsBefore: 8, neighborsAfter: 9, neighborStep: 12,
        previewColumns: 5, previewPage: 15, topG: 12, hierExpand: 6,
    },
    medium: {
        label: "Medium",
        columns: 5, neighborsBefore: 7, neighborsAfter: 7, neighborStep: 10,
        previewColumns: 4, previewPage: 12, topG: 10, hierExpand: 5,
    },
    large: {
        label: "Large",
        columns: 4, neighborsBefore: 5, neighborsAfter: 6, neighborStep: 8,
        previewColumns: 3, previewPage: 12, topG: 8, hierExpand: 4,
    },
};

export const TILE_SIZE_KEYS = ["small", "medium", "large"];

const STORAGE_KEY = "routing101_settings";

function clampZoom(v) {
    const n = Number(v);
    if (!isFinite(n)) return SETTINGS_DEFAULTS.hoverZoom;
    // Snap to the slider's own grid so a hand-edited/older stored value can't
    // leave the control sitting between two steps.
    const snapped = Math.round((n - HOVER_ZOOM_MIN) / HOVER_ZOOM_STEP) * HOVER_ZOOM_STEP + HOVER_ZOOM_MIN;
    return Math.min(HOVER_ZOOM_MAX, Math.max(HOVER_ZOOM_MIN, Math.round(snapped * 10) / 10));
}

function normalize(raw) {
    return {
        hoverZoom: clampZoom(raw?.hoverZoom ?? SETTINGS_DEFAULTS.hoverZoom),
        tileSize: TILE_SIZES[raw?.tileSize] ? raw.tileSize : SETTINGS_DEFAULTS.tileSize,
        groupByVideo: Boolean(raw?.groupByVideo ?? SETTINGS_DEFAULTS.groupByVideo),
        showFullText: Boolean(raw?.showFullText ?? SETTINGS_DEFAULTS.showFullText),
    };
}

function load() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (raw) return normalize(JSON.parse(raw));
    } catch (e) { /* corrupt/old value -- fall through to defaults */ }
    return { ...SETTINGS_DEFAULTS };
}

export const settings = load();

/** The active tile-size spec (column counts, page sizes, Top-G default). */
export function tile() {
    return TILE_SIZES[settings.tileSize] || TILE_SIZES.medium;
}

// ---------------------------------------------------------------------------
// Query chunking -- the one setting in this dialog that does NOT live here.
// SigLIP2's text tower reads at most 64 tokens, so a longer query has to be
// split; the strategy decides what happens to the pieces (backend/models.py).
// It's backend state, not a browser preference: the splitting and the
// embedding both happen in the backend process, and two tabs on the same port
// share one value. So this module only carries the labels and a cached copy
// of what the backend last reported -- the dialog re-reads /api/settings on
// open rather than trusting the cache.
// ---------------------------------------------------------------------------

export const QUERY_CHUNK_LABELS = {
    truncate: {
        label: "Truncate",
        title: "Keep only the first 64 tokens of the query and drop the rest. What the app did before chunking existed.",
    },
    mean_chunks: {
        label: "Average",
        title: "Split into sentence-aligned chunks, embed each, and search with their average. A soft AND: a result has to look somewhat like the whole query.",
    },
    chunks_separate: {
        label: "Per chunk",
        title: "Split into sentence-aligned chunks, search with each one, and RRF-fuse the rankings. A result is rewarded for ranking well against several chunks. Scores shown are RRF scores, not similarities.",
    },
};

export const QUERY_CHUNK_DEFAULT = "chunks_separate";

// Last value the backend reported, so the dialog has something to draw
// before its own fetch resolves. Never written to localStorage.
export const queryChunk = { strategy: QUERY_CHUNK_DEFAULT };

export function setQueryChunkCache(strategy) {
    if (QUERY_CHUNK_LABELS[strategy]) queryChunk.strategy = strategy;
    return queryChunk.strategy;
}

// How the settings dialog should present the one shared group-by toggle for
// the signal that's currently mounted -- Hierarchy/TRAKE don't offer it at
// all, and Summary relabels it (it groups by collection, not video: one
// toggle relabelled, never a second checkbox, same as ui/app.py). Signals set
// this from mount()/unmount() instead of reaching into the sidebar DOM, which
// is where these two rules lived before the checkbox moved into Settings.
export const GROUP_BY_DEFAULT_LABEL = "Group by video";
export const groupByUi = { visible: true, label: GROUP_BY_DEFAULT_LABEL };

export function setGroupByUi({ visible = true, label = GROUP_BY_DEFAULT_LABEL } = {}) {
    groupByUi.visible = visible;
    groupByUi.label = label;
}

// Everything layout-ish is driven through CSS custom properties on :root, so
// a size/zoom change takes effect on already-rendered grids without a
// re-render -- style.css consumes --hover-zoom/--grid-columns/--preview-columns.
export function applySettings() {
    const root = document.documentElement;
    root.style.setProperty("--hover-zoom", String(settings.hoverZoom));
    root.style.setProperty("--grid-columns", String(tile().columns));
    root.style.setProperty("--preview-columns", String(tile().previewColumns));
}

export function saveSettings(next) {
    Object.assign(settings, normalize({ ...settings, ...next }));
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch (e) { /* private mode / quota -- keep the in-memory value anyway */ }
    applySettings();
}

// The Export CSV tab is a separate window sharing these preferences, so a
// Save in the search tab repaints it live instead of waiting for a reload.
// (Only fires in *other* tabs -- the saving tab already applied above.)
window.addEventListener("storage", (e) => {
    if (e.key !== STORAGE_KEY) return;
    Object.assign(settings, load());
    applySettings();
});

applySettings();
