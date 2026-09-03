# Routing101

Multi-signal text-to-keyframe retrieval over the AIC video corpus (873
videos / ~177k keyframes), plus an AIC-submission CSV export flow. One
FastAPI process serves both the JSON API and the static frontend.

It runs against one of three SigLIP2 embedding profiles -- 768-dim
(default), 1152-dim or 1536-dim -- chosen at launch; see [Embedding profiles](#embedding-profiles).

This file is a **build-and-run guide** — pick your track below and follow
it top to bottom. For how the system actually works (layout, signals,
export flow, etc.), see [`ARCHITECTURE.md`](ARCHITECTURE.md) instead.

- [Prerequisites](#prerequisites)
- [Embedding profiles](#embedding-profiles)
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

  Embeddings for the **1152-dim** and **1536-dim** profiles — only needed
  if you want to run one of them; the app is fully usable without them.
  Identical layout, one folder per dimension:

  ```
  AICDataExtracted/1152embed/1152keyframe/          frame embeddings
  AICDataExtracted/1152embed/1152transcript/        ASR-segment embeddings + .csv
  AICDataExtracted/1152embed/1152caption/           caption embeddings + .csv
  AICDataExtracted/1152embed/1152summary/           summary embeddings (chunked) + .csv

  AICDataExtracted/1536embed/1536keyframe/          frame embeddings
  AICDataExtracted/1536embed/1536transcript/        ASR-segment embeddings + .csv
  AICDataExtracted/1536embed/1536caption/           caption embeddings + .csv
  AICDataExtracted/1536embed/1536summary/           summary embeddings (chunked) + .csv
  ```

  `summary_embed/` and `index/` are write targets too (the app creates/
  fills them itself on first run) -- just make sure the parent directory
  is writable. `1152embed/` and `1536embed/` are read-only: those
  embeddings come from the upstream pipeline, the app never generates them.

- **Elasticsearch 8.x**, reachable at boot. This is not optional even if
  you don't care about fuzzy text search: `backend/main.py`'s startup
  calls `ensure_all_fuzzy_indices()` with no try/except around it, so an
  unreachable Elasticsearch **crashes the whole app on launch**, not just
  the fuzzy legs. (Once the four indices exist and the app is up, a
  *later* ES outage does degrade gracefully -- empty results + a warning,
  per signal. It's only the boot-time existence check that's unguarded.)
  How you get Elasticsearch differs by track, see below.

## Embedding profiles

The app can run against any one of three SigLIP2 checkpoints. You pick one
**before the process starts** -- there is no switch inside the UI:

| | `768` (default) | `1152` | `1536` |
|---|---|---|---|
| Checkpoint | `siglip2-base-patch16-384` | `siglip2-so400m-patch14-384` | `siglip2-giant-opt-patch16-384` |
| Embeddings | `siglib_embed/`, `transcript_embed/`, `caption_embed/`, `summary_embed/` | `1152embed/1152*/` | `1536embed/1536*/` |
| FAISS indices | `index/routing101_*` | `index/1152/routing101_*` | `index/1536/routing101_*` |
| Port | 8000 | 8001 | 8002 |
| Model download | ~1.4 GB | ~4.2 GB | ~7 GB |
| RAM (measured) | ~2.8 GB | ~4-5 GB | ~9.4 GB |

Selected with the `R101_EMBED` environment variable (`768`, `1152` or
`1536`, default `768`), which `backend/config.py` reads once at import. On
Windows the launcher scripts set it for you -- **`run_768.bat`**,
**`run_1152.bat`** and **`run_1536.bat`**, all in the repo. On any other
platform, set the env var yourself; see Track A step 4.

Each profile is its own process on its own port, so you can run two at once
and compare the same query in two tabs -- the coloured pill next to the page
title says which profile that tab is talking to. Running all three at once
does not fit in 32 GB; stop one before starting a third. Everything that
isn't an embedding is shared: one Elasticsearch container, the same
thumbnails/video, the same export flow. No profile can clobber another's
indices.

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
   you don't have the `1152embed/` or `1536embed/` folders, leave those
   entries alone -- an unused profile is never touched.

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
   run_1536.bat     1536-dim profile -> http://localhost:8002/app/
   ```
   Run two if you want them side by side (all three at once won't fit in
   32 GB). `stop_routing101.bat 8000` (or `8001`, `8002`) stops one without
   touching Elasticsearch, so the next launch stays fast. All three
   launchers are four-line wrappers over `_run_common.bat`, which holds the
   shared bootstrap -- edit that one, not the three.

   Leave the launcher window open while you work; it is the live server
   log. Closing it (or Ctrl+C, then Y) stops the app.

   On any other platform, or to skip the Docker/browser handling, it's the
   env var plus a port:
   ```
   uvicorn backend.main:app --reload                              # 768, :8000
   R101_EMBED=1152 uvicorn backend.main:app --reload --port 8001  # 1152, :8001
   R101_EMBED=1536 uvicorn backend.main:app --reload --port 8002  # 1536, :8002
   ```
   First boot loads the SigLIP2 model, builds the FAISS indices, and
   bulk-indexes the four ES fuzzy indices (only the first time -- see
   step 3's volume note) -- expect 15-60s depending on disk speed, not
   instant. Watch for `[startup] all signals ready` in the console; the
   server *does* accept connections a little before that line prints (see
   [Troubleshooting](#troubleshooting) on log buffering) but search
   requests will just queue until it's actually done.

   **The 1152 and 1536 profiles' first runs are much slower**: they
   download ~4.2 GB and ~7 GB of model weights respectively and write their
   FAISS indices to `index/1152/` or `index/1536/` from scratch, which took
   several minutes on the dev machine. Every run after that loads in ~20s
   like the 768 one. The startup log prints the profile and per-index vector
   counts, so you can confirm what you actually got:
   ```
   [startup] profile=1152 dim=1152 model=google/siglip2-so400m-patch14-384
   [startup]   177321 frames over 873 videos
   [startup]   2501 chunks over 785 videos
   ```

5. Open **http://localhost:8000/app/** (or `:8001` / `:8002` for the 1152
   and 1536 profiles).
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
   The 1152 and 1536 profiles only ever read their summary embeddings, so
   a read-only mount is fine there.

3. **Install dependencies** (Kaggle images already have `numpy`/`pandas`/
   `torch`; this just fills in what's missing):
   ```
   !pip install -q fastapi uvicorn python-multipart cachetools \
       faiss-cpu elasticsearch
   ```

4. **Get Elasticsearch running.** Kaggle notebooks don't run a Docker
   daemon, so the `docker run` command from Track A is out. Two options:
   - **Simplest: point `ES_HOST` at a hosted Elasticsearch** (Elastic
     Cloud has a free trial) instead of `http://localhost:9200`. No
     process-management inside the notebook at all.
   - **Self-contained: run the ES binary directly in the notebook.**
     Elasticsearch refuses to run as root, and Kaggle notebooks run as
     root by default, so you need an unprivileged user first:
     ```
     !wget -q https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.15.0-linux-x86_64.tar.gz
     !tar -xzf elasticsearch-8.15.0-linux-x86_64.tar.gz
     !useradd -m esuser && chown -R esuser:esuser elasticsearch-8.15.0
     import subprocess
     subprocess.Popen(
         ["su", "esuser", "-c",
          "elasticsearch-8.15.0/bin/elasticsearch "
          "-E discovery.type=single-node -E xpack.security.enabled=false "
          "-E xpack.ml.enabled=false"],
     )
     ```
     Then poll `curl http://localhost:9200` in a loop until it responds
     (same as Track A step 3) before starting the app -- and remember a
     Kaggle session is ephemeral, so this bulk-indexes from CSV fresh
     every single session (no persistent volume like Track A's Docker
     one), which adds to your startup time.

5. **Run the app as a background process**, same command as local, just
   bound to all interfaces:
   ```
   !nohup uvicorn backend.main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
   ```
   Prefix `R101_EMBED=1152` (or `R101_EMBED=1536`) to run one of the other
   profiles instead. Attach the matching `1152embed/` or `1536embed/`
   folders to your dataset first, and expect the longer first boot -- a
   ~4.2 GB or ~7 GB model download plus a from-scratch FAISS build, every
   session, since Kaggle sessions are ephemeral.
   Running it this way (a real background OS process via `!`, not calling
   `uvicorn.run()` inline in a cell) sidesteps Jupyter's already-running
   event loop entirely -- no `nest_asyncio` dance needed. Poll
   `!curl http://localhost:8000/app/` until it 200s before moving on, same
   readiness caveat as Track A step 4.

6. **Expose the port to your actual browser.** Kaggle gives you no public
   URL for an arbitrary local port, so tunnel it -- `pyngrok` is the
   standard fix:
   ```
   !pip install -q pyngrok
   from pyngrok import ngrok
   print(ngrok.connect(8000, "http"))
   ```
   Open the printed `https://....ngrok-free.app` URL -- `/app/` etc. all
   work the same, everything the frontend fetches is a same-origin
   relative path (`/api/...`), so there's no CORS wrinkle from tunneling.

7. **Optional: turn on a GPU** (Settings → Accelerator → GPU T4 x2 or
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
- The pill next to the page title reads `768d`, `1152d` or `1536d` and
  matches the port you opened. If you're comparing profiles in two tabs,
  check this before trusting either tab's results -- they look identical
  otherwise.

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
- **`R101_EMBED='...' -- expected one of ['1152', '1536', '768']` and the process
  exits.** Exactly what it says: the env var is set to something that
  isn't a profile name. Unset it for the default.
- **A FAISS dimension assertion at startup, or every search 500s on one
  profile.** A profile is reading another profile's vectors -- almost
  always a `backend/config.py` edit that pointed two profiles at the same
  `summary_embed_dir` or the same `index_sub`. Each profile needs its own
  of both; delete the affected `index/` subtree and let it rebuild.
- **A profile starts but a signal covers fewer videos than you expect.**
  Check the per-index counts in the startup log against the 873-video
  corpus. The 1152 profile's ASR embeddings are incomplete upstream around
  L25 (773 videos indexed as of this writing), which is a data gap, not a
  bug -- its other signals are complete, and 1536 was embedded after that
  gap was filled.
