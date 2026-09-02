# Routing101

Multi-signal text-to-keyframe retrieval over the AIC video corpus (873
videos / ~177k keyframes), plus an AIC-submission CSV export flow. One
FastAPI process serves both the JSON API and the static frontend.

It runs against either of two SigLIP2 embedding profiles -- 768-dim
(default) or 1152-dim -- chosen at launch; see [Embedding profiles](#embedding-profiles-768-vs-1152).

This file is a **build-and-run guide** — pick your track below and follow
it top to bottom. For how the system actually works (layout, signals,
export flow, etc.), see [`ARCHITECTURE.md`](ARCHITECTURE.md) instead.

- [Prerequisites](#prerequisites)
- [Embedding profiles: 768 vs 1152](#embedding-profiles-768-vs-1152)
- [Track A — Run locally](#track-a--run-locally)
- [Track B — Run on Kaggle](#track-b--run-on-kaggle)
- [Verifying it's working](#verifying-its-working)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- **Python** 3.11+ (developed against 3.14; nothing in `requirements.txt`
  pins a floor, but don't go below 3.10 -- `torch>=2.8.0` won't install).
- **The data.** This repo ships no data at all -- get the `AICData` +
  `AICDataExtracted` folders from the team (shared drive/bucket, ask
  whoever last ran the pipeline). Both tracks below need these, just
  staged in different places. The exact subfolders you need:

  Shared by both embedding profiles (see the next section):

  ```
  AICDataExtracted/transcripts/                     raw ASR segments (bulk-indexed into ES)
  AICDataExtracted/captions/                        raw frame captions (bulk-indexed into ES)
  AICDataExtracted/ocr/                             per-frame OCR text (bulk-indexed into ES)
  AICDataExtracted/summaries/                       one-paragraph video summaries (raw text)
  AICDataExtracted/filtered_object/ (+class_vocab.csv)  per-frame OD detections + vocabulary
  AICData/map-keyframes/*.csv                       per-frame timestamps + native frame_idx
  AICData/keyframes/{video_id}/{n:03d}.jpg           thumbnails
  AICData/video/{video_id}.mp4                       source video (playback dialogs)
  ```

  Embeddings for the **768-dim** profile (the default):

  ```
  AICDataExtracted/siglib_embed/*.npy               frame embeddings
  AICDataExtracted/transcript_embed/                ASR-segment embeddings + .csv
  AICDataExtracted/caption_embed/                   caption embeddings + .csv
  AICDataExtracted/summary_embed/                   summary embeddings (app fills this itself)
  ```

  Embeddings for the **1152-dim** profile — only needed if you want to run
  it; the app is fully usable without them:

  ```
  AICDataExtracted/1152embed/1152keyframe/          frame embeddings
  AICDataExtracted/1152embed/1152transcript/        ASR-segment embeddings + .csv
  AICDataExtracted/1152embed/1152caption/           caption embeddings + .csv
  AICDataExtracted/1152embed/1152summary/           summary embeddings (chunked) + .csv
  ```

  `summary_embed/` and `index/` are write targets too (the app creates/
  fills them itself on first run) -- just make sure the parent directory
  is writable. `1152embed/` is read-only: those embeddings come from the
  upstream pipeline, the app never generates them.

- **Elasticsearch 8.x**, reachable at boot. This is not optional even if
  you don't care about fuzzy text search: `backend/main.py`'s startup
  calls `ensure_all_fuzzy_indices()` with no try/except around it, so an
  unreachable Elasticsearch **crashes the whole app on launch**, not just
  the fuzzy legs. (Once the four indices exist and the app is up, a
  *later* ES outage does degrade gracefully -- empty results + a warning,
  per signal. It's only the boot-time existence check that's unguarded.)
  How you get Elasticsearch differs by track, see below.

## Embedding profiles: 768 vs 1152

The app can run against either of two SigLIP2 checkpoints. You pick one
**before the process starts** -- there is no switch inside the UI:

| | `768` (default) | `1152` |
|---|---|---|
| Checkpoint | `siglip2-base-patch16-384` | `siglip2-so400m-patch14-384` |
| Embeddings | `siglib_embed/`, `transcript_embed/`, `caption_embed/`, `summary_embed/` | `1152embed/1152*/` |
| FAISS indices | `index/routing101_*` | `index/1152/routing101_*` |
| Port | 8000 | 8001 |
| Model download | ~1.4 GB | ~4.2 GB |
| RAM (measured) | ~2.8 GB | ~4-5 GB |

Selected with the `R101_EMBED` environment variable (`768` or `1152`,
default `768`), which `backend/config.py` reads once at import. On Windows
the launcher scripts set it for you -- **`run_768.bat`** and
**`run_1152.bat`**, both in the repo. On any other platform, set the env
var yourself; see Track A step 4.

The two profiles are separate processes on separate ports, so you can run
both at once and compare the same query in two tabs -- the coloured pill next to the
page title says which profile that tab is talking to. Everything that
isn't an embedding is shared: one Elasticsearch container, the same
thumbnails/video, the same export flow. Neither profile can clobber the
other's indices.

**If you only want the app working, ignore all of this** -- do nothing and
you get the 768 profile, exactly as before.

## Track A — Run locally

1. **Clone and install:**
   ```
   git clone <this repo> && cd Routing101
   pip install -r requirements.txt
   ```

2. **Point the app at your data.** All data paths are hardcoded module
   constants (single-team local scaffold, no env-file layer) -- edit
   `backend/config.py`'s every `*_DIR`/`*_GLOB` constant near the top
   (`FRAME_SIGLIP2_GLOB`, `ASR_EMBED_DIR`, `TRANSCRIPTS_DIR`, ...,
   `MAP_KEYFRAMES_DIR`, `THUMBNAIL_ROOT`, `VIDEO_DIR`) to match wherever
   you staged the folders above. They're all absolute paths under one root
   today (`D:/University/Summ26/AICData*`) -- if you mirror that same
   layout, it's a one-line prefix change per platform, not per constant.

   Most of them now live in `config.py`'s `_PROFILES` table (one entry per
   embedding profile) rather than as flat module constants; `_EXTRACTED`
   just above it is the single root prefix those entries hang off, so
   retargeting a whole machine is usually one edit to that one line. If
   you don't have the `1152embed/` folders, leave the `"1152"` entry
   alone -- an unused profile is never touched.

3. **Start Elasticsearch** (Docker):
   ```
   docker run -d --name es -p 9200:9200 \
       -e "discovery.type=single-node" \
       -e "xpack.security.enabled=false" \
       -e "xpack.ml.enabled=false" \
       -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
       -v es-data:/usr/share/elasticsearch/data \
       docker.elastic.co/elasticsearch/elasticsearch:8.15.0
   ```
   The `-v es-data:...` volume matters: without it, `docker rm`/recreate
   loses all four fuzzy indices and the next launch re-bulks everything
   from CSV. With the volume, `ensure_*_fuzzy_index()` in
   `backend/es_indexing.py` checks `es.indices.exists(...)` up front and
   only (re)indexes an index that doesn't exist yet -- to force a rebuild
   after changing source data, delete that one index (e.g.
   `curl -X DELETE localhost:9200/caption_frames`) rather than the whole
   container/volume.

   Wait for it to actually be up before the next step:
   ```
   curl http://localhost:9200   # should return a JSON cluster info blob, not a connection error
   ```

4. **Run the app.** On Windows, the launcher scripts do steps 3-5 for you
   -- start Docker if it isn't running, start (or create) the `es`
   container, wait for it, launch the backend, open your browser:
   ```
   run_768.bat      768-dim profile  -> http://localhost:8000/app/
   run_1152.bat     1152-dim profile -> http://localhost:8001/app/
   ```
   Run both if you want them side by side. `stop_routing101.bat 8000` (or
   `8001`) stops one without touching Elasticsearch, so the next launch
   stays fast. Both launchers are four-line wrappers over
   `_run_common.bat`, which holds the shared bootstrap -- edit that one,
   not the two.

   Leave the launcher window open while you work; it is the live server
   log. Closing it (or Ctrl+C, then Y) stops the app.

   On any other platform, or to skip the Docker/browser handling, it's the
   env var plus a port:
   ```
   uvicorn backend.main:app --reload                              # 768, :8000
   R101_EMBED=1152 uvicorn backend.main:app --reload --port 8001  # 1152, :8001
   ```
   First boot loads the SigLIP2 model, builds the FAISS indices, and
   bulk-indexes the four ES fuzzy indices (only the first time -- see
   step 3's volume note) -- expect 15-60s depending on disk speed, not
   instant. Watch for `[startup] all signals ready` in the console; the
   server *does* accept connections a little before that line prints (see
   [Troubleshooting](#troubleshooting) on log buffering) but search
   requests will just queue until it's actually done.

   **The 1152 profile's first run is much slower**: it downloads ~4.2 GB
   of model weights and writes its FAISS indices to `index/1152/` from
   scratch, which took several minutes on the dev machine. Every run after
   that loads in ~20s like the 768 one. The startup log prints the profile
   and per-index vector counts, so you can confirm what you actually got:
   ```
   [startup] profile=1152 dim=1152 model=google/siglip2-so400m-patch14-384
   [startup]   177321 frames over 873 videos
   [startup]   2501 chunks over 785 videos
   ```

5. Open **http://localhost:8000/app/** (or `:8001` for the 1152 profile).
   The pill next to the page title confirms which profile you're on.

`--reload` is convenient but its file-watcher isn't fully reliable in
every environment -- if you edit a `backend/*.py` file and don't see the
change take effect, kill the process and restart manually rather than
trusting it blindly. Frontend edits (`frontend/**`) never need a restart
at all, they're re-served fresh on every request.

## Track B — Run on Kaggle

The core app doesn't change for Kaggle -- same FastAPI process, same
`uvicorn` command -- but three things about the *environment* do: no
persistent disk with your data already on it, no Docker daemon for
Elasticsearch, and no public port for a browser to reach.

### Quick Start with Pre-built Kaggle Notebook

For the simplest and most robust setup, use the included **`kaggle_routing101.ipynb`** notebook:
1. Upload [kaggle_routing101.ipynb](file:///d:/StudioProjects/Routing101/kaggle_routing101.ipynb) to a Kaggle Notebook.
2. Set **Accelerator**: `GPU T4 x2` (or P100) and turn **Internet: ON**.
3. Attach your datasets:
   - `aic2026-dataset` (`Keyframes`, `Videos`, `map-keyframes`)
   - `rrqbundle` (`siglib_embed`, `ocr`, `captions`, `summaries`, `filtered_object`, etc.)
4. Run all cells in sequence (Step 1 -> Step 5).
   - **Step 1**: Clones repo & installs dependencies.
   - **Step 2**: Starts Elasticsearch 8.11 daemon cleanly with JVM heap limits.
   - **Step 3**: Deep pruned auto-discovery dynamically maps dataset paths.
   - **Step 4**: Diagnostics check (Elasticsearch + SigLIP2 vector search).
   - **Step 5**: Launches FastAPI + Uvicorn with Cloudflare Tunnel (no token required).

---

### Manual Setup Steps (if running step-by-step)

1. **Get the data onto Kaggle.** Package `AICData`/`AICDataExtracted` as
   a Kaggle Dataset (zip it and use "New Dataset", or "Add Data" in your
   notebook if a teammate already published one for the team) and attach
   it to your notebook. It mounts read-only under
   `/kaggle/input/<dataset-slug>/...`.

2. **Point the app at Kaggle's paths.** Same file as Track A step 2
   (`backend/config.py`), just pointed at
   `/kaggle/input/<dataset-slug>/...` instead of the `D:/...` paths --
   in practice, repoint `_EXTRACTED` and the `AICData` constants.
   `INDEX_DIR` (under `REPO_ROOT`, i.e. wherever you `!git clone`'d the
   repo inside the notebook) is fine as-is -- it's a write target the app
   creates itself, and `/kaggle/working/` has room.

   `summary_embed_dir` needs care: on the 768 profile the app *writes*
   into it on first run, but Kaggle input datasets mount **read-only**, so
   point it at a writable path (e.g. under `/kaggle/working/`) and let it
   fill, or ship the summary embeddings in your dataset and point at them.
   The 1152 profile only ever reads its summary embeddings, so a read-only
   mount is fine there.

3. **Install dependencies** (Kaggle images already have `numpy`/`pandas`/
   `torch`; this just fills in what's missing):
   ```bash
   !pip install -q fastapi uvicorn python-multipart cachetools \
       "elasticsearch>=8.11,<8.13" faiss-cpu pycloudflared
   ```

4. **Get Elasticsearch running.** Kaggle notebooks don't run a Docker
   daemon, so the `docker run` command from Track A is out. Two options:
   - **Simplest: point `ES_HOST` at a hosted Elasticsearch** (Elastic
     Cloud has a free trial) instead of `http://127.0.0.1:9200`.
   - **Self-contained: run the ES binary directly in the notebook:**
     ```bash
     !wget -q https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.11.0-linux-x86_64.tar.gz
     !tar -xzf elasticsearch-8.11.0-linux-x86_64.tar.gz -C /opt/
     !useradd -m -s /bin/bash esuser 2>/dev/null || true
     !chown -R esuser:esuser /opt/elasticsearch-8.11.0
     !touch /kaggle/working/elasticsearch.log && chmod 666 /kaggle/working/elasticsearch.log
     ```
     Run via Python `subprocess.Popen` with `start_new_session=True` under `esuser`.

5. **Run the app and tunnel to your browser**:
   Start Uvicorn in the background and expose via `cloudflared tunnel --url http://127.0.0.1:8000`.
   Prefix `R101_EMBED=1152` if running the 1152-dim profile instead.

6. **Optional: turn on a GPU** (Settings → Accelerator → GPU T4 x2 or
   P100) before running -- `backend/models.py` auto-detects
   `torch.cuda.is_available()`, no code change needed, and it meaningfully
   speeds up SigLIP2 inference over CPU-only.

## Verifying it's working

Once you're on `/app/`:
- The left sidebar's signal icons should switch panels; typing a query
  and hitting enter should return keyframe results.
- Any fuzzy-text signal (ASR/Caption/OCR/Summary) returning results
  (not just a warning banner) confirms Elasticsearch is actually wired up
  correctly, not just running.
- A result card's ▶ (playback) and ★ (export) buttons should both open
  without errors -- ▶ needs `AICData/video/*.mp4` + `map-keyframes`
  resolvable, ★ needs `map-keyframes` too.
- The pill next to the page title reads `768d` or `1152d` and matches the
  port you opened. If you're comparing profiles in two tabs, check this
  before trusting either tab's results -- the two look identical otherwise.

## Troubleshooting

- **"Loading weights: 100%" then nothing for a while, then a burst of
  `[startup] ...` lines all at once, `all signals ready` right at the
  end.** Normal -- Python fully buffers `print()` output when it's not
  writing to a real terminal (redirected to a log file, `nohup`, etc.),
  so those lines were already printed, they just hadn't flushed yet. The
  process itself is not stuck; if you want ground truth, poll an actual
  endpoint (`curl localhost:8000/app/`) rather than staring at the log.
- **Two things end up listening on the same port after a restart.**
  `--reload`'s file-watcher can silently miss an edit (only reload once,
  then stop noticing further changes), and a killed process can leave an
  orphaned multiprocessing worker behind that keeps holding memory (and,
  confusingly, `netstat` can keep showing the old dead PID against that
  port for a bit). If a restart is behaving strangely, check
  `tasklist`/`ps` for more `python` processes than you expect and kill
  the stragglers, not just the one you started last.
- **A browser tab keeps showing old UI/behavior after you *know* the
  server has the new code.** `frontend/` is served with
  `Cache-Control: no-cache` (forces revalidation, not `no-store` -- an
  unchanged file still 304s), but that only applies going forward from
  when this was added; a tab that already cached a file under the old
  (implicit, heuristic-freshness) caching behavior won't retroactively
  revalidate on its own. One hard refresh (Ctrl+Shift+R) fixes that tab
  for good; a brand-new tab is never affected.
- **The whole app fails to start with a connection error mentioning
  Elasticsearch.** Elasticsearch isn't reachable at `ES_HOST` -- see the
  Prerequisites section above, this is a hard requirement at boot, not a
  soft one.
- **`R101_EMBED='...' -- expected one of ['1152', '768']` and the process
  exits.** Exactly what it says: the env var is set to something that
  isn't a profile name. Unset it for the default.
- **A FAISS dimension assertion at startup, or every search 500s on one
  profile.** A profile is reading another profile's vectors -- almost
  always a `backend/config.py` edit that pointed two profiles at the same
  `summary_embed_dir` or the same `index_sub`. Each profile needs its own
  of both; delete the affected `index/` subtree and let it rebuild.
- **The 1152 profile starts but a signal covers fewer videos than you
  expect.** Check the per-index counts in the startup log against the
  873-video corpus. Its ASR embeddings are incomplete upstream around L25
  (773 videos indexed as of this writing), which is a data gap, not a bug
  -- the other signals are complete.
