# Routing101

Multi-signal text-to-keyframe retrieval over the AIC video corpus (873
videos / ~177k keyframes), plus an AIC-submission CSV export flow. One
FastAPI process serves both the JSON API and the static frontend.

This file is a **build-and-run guide** — pick your track below and follow
it top to bottom. For how the system actually works (layout, signals,
export flow, etc.), see [`ARCHITECTURE.md`](ARCHITECTURE.md) instead.

- [Prerequisites](#prerequisites)
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

  ```
  AICDataExtracted/siglib_embed/*.npy              SigLIP2 frame embeddings (768-d)
  AICDataExtracted/transcript_embed/                SigLIP2 ASR-segment embeddings + .csv
  AICDataExtracted/transcripts/                     raw ASR segments (bulk-indexed into ES)
  AICDataExtracted/caption_embed/                   SigLIP2 caption embeddings + .csv
  AICDataExtracted/captions/                        raw frame captions (bulk-indexed into ES)
  AICDataExtracted/ocr/                             per-frame OCR text (bulk-indexed into ES)
  AICDataExtracted/summaries/ + summary_embed/       one-paragraph video summaries
  AICDataExtracted/filtered_object/ (+class_vocab.csv)  per-frame OD detections + vocabulary
  AICData/clip-features-32/*.npy                    CLIP ViT-B/32 frame embeddings (512-d, fp16)
  AICData/map-keyframes/*.csv                       per-frame timestamps + native frame_idx
  AICData/keyframes/{video_id}/{n:03d}.jpg           thumbnails
  AICData/video/{video_id}.mp4                       source video (playback dialogs)
  ```

  `summary_embed/` and `index/` are write targets too (the app creates/
  fills them itself on first run) -- just make sure the parent directory
  is writable.

- **Elasticsearch 8.x**, reachable at boot. This is not optional even if
  you don't care about fuzzy text search: `backend/main.py`'s startup
  calls `ensure_all_fuzzy_indices()` with no try/except around it, so an
  unreachable Elasticsearch **crashes the whole app on launch**, not just
  the fuzzy legs. (Once the four indices exist and the app is up, a
  *later* ES outage does degrade gracefully -- empty results + a warning,
  per signal. It's only the boot-time existence check that's unguarded.)
  How you get Elasticsearch differs by track, see below.

## Track A — Run locally

1. **Clone and install:**
   ```
   git clone <this repo> && cd Routing101
   pip install -r requirements.txt
   ```

2. **Point the app at your data.** All data paths are hardcoded module
   constants (single-team local scaffold, no env-file layer) -- edit
   these two files to match wherever you staged the folders above:
   - `backend/config.py` -- every `*_DIR`/`*_GLOB` constant near the top
     (`FRAME_SIGLIP2_GLOB`, `ASR_EMBED_DIR`, `TRANSCRIPTS_DIR`, ...,
     `MAP_KEYFRAMES_DIR`, `THUMBNAIL_ROOT`, `VIDEO_DIR`). They're all
     absolute paths under one root today (`D:/University/Summ26/AICData*`)
     -- if you mirror that same layout, it's a one-line prefix change per
     platform, not per constant.
   - `pipeline/config.py` -- just a model name, nothing to change here
     unless you're swapping the CLIP text tower.

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

4. **Run the app:**
   ```
   uvicorn backend.main:app --reload
   ```
   First boot loads the SigLIP2 model, builds the FAISS indices in
   memory, and bulk-indexes the four ES fuzzy indices (only the first
   time -- see step 3's volume note) -- expect 15-60s depending on disk
   speed, not instant. Watch for `[startup] all signals ready` in the
   console; the server *does* accept connections a little before that
   line prints (see [Troubleshooting](#troubleshooting) on log buffering)
   but search requests will just queue until it's actually done.

5. Open **http://localhost:8000/app/**.

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

2. **Point the app at Kaggle's paths.** Same two files as Track A step 2
   (`backend/config.py`, `pipeline/config.py`), just pointed at
   `/kaggle/input/<dataset-slug>/...` instead of the `D:/...` paths.
   `INDEX_DIR`/`SUMMARY_EMBED_DIR` (under `REPO_ROOT`, i.e. wherever you
   `!git clone`'d the repo inside the notebook) are fine as-is -- they're
   write targets the app creates itself, `/kaggle/working/` has room.

3. **Install dependencies** (Kaggle images already have `numpy`/`pandas`/
   `torch`; this just fills in what's missing):
   ```
   !pip install -q fastapi uvicorn python-multipart cachetools \
       faiss-cpu elasticsearch multilingual-clip timm
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
   speeds up SigLIP2/CLIP inference over CPU-only.

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
