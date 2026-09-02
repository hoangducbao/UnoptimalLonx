@echo off
rem run_1152.bat -- launches Routing101 on the 1152-dim embedding profile
rem (google/siglip2-so400m-patch14-384) at http://localhost:8001.
rem
rem Its FAISS indices live in index/1152/routing101_* -- separate files from
rem the 768 profile's, so neither can clobber the other. The very first run
rem on a machine spends several minutes building them (and downloads ~4.2GB
rem of model weights); every run after that just loads, in about 20 seconds.
rem ~4-5GB resident, more than the 768 profile.
rem
rem Run run_768.bat alongside this for the 768-dim profile on :8000 -- the
rem two are separate processes sharing one Elasticsearch container and the
rem same media files, so you can compare the same query in two tabs. The
rem header pill in the UI says which profile a tab is talking to.
rem
rem Leave this window open while you work -- it shows the live server log.
rem Close it (or Ctrl+C, then press Y) to stop the app, or run
rem stop_routing101.bat 8001 from elsewhere.

setlocal
set "PROFILE=1152"
set "PORT=8001"
call "%~dp0_run_common.bat"
endlocal
