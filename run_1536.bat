@echo off
rem run_1536.bat -- launches Routing101 on the 1536-dim embedding profile
rem (google/siglip2-giant-opt-patch16-384) at http://localhost:8002.
rem
rem Its FAISS indices live in index/1536/routing101_* -- separate files from
rem the 768 and 1152 profiles', so none of the three can clobber another.
rem The very first run on a machine spends several minutes building them,
rem and downloads ~7GB of model weights: the giant tower is a bigger
rem download than 1152's ~4.2GB and it is not already in the HF cache.
rem Every run after that just loads, in well under a minute.
rem ~9.4GB resident (and ~20GB private commit), by far the heaviest of the
rem three profiles -- pair it with run_768.bat, not run_1152.bat, on 32GB.
rem
rem Run run_768.bat (:8000) or run_1152.bat (:8001) alongside this to compare
rem the same query in two tabs -- separate processes sharing one Elasticsearch
rem container and the same media files. The header pill in the UI says which
rem profile a tab is talking to. Two profiles at once fit comfortably; all
rem three do not, so stop one before starting the third.
rem
rem Leave this window open while you work -- it shows the live server log.
rem Close it (or Ctrl+C, then press Y) to stop the app, or run
rem stop_routing101.bat 8002 from elsewhere.

setlocal
set "PROFILE=1536"
set "PORT=8002"
call "%~dp0_run_common.bat"
endlocal
