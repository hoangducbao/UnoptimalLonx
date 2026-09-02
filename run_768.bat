@echo off
rem run_768.bat -- launches Routing101 on the 768-dim embedding profile
rem (google/siglip2-base-patch16-384) at http://localhost:8000.
rem
rem This is the original profile: its FAISS indices live in index/routing101_*
rem and are already built, so startup is just a load. ~2.8GB resident.
rem
rem Run run_1152.bat alongside this for the 1152-dim profile on :8001 -- the
rem two are separate processes sharing one Elasticsearch container and the
rem same media files, so you can compare the same query in two tabs. The
rem header pill in the UI says which profile a tab is talking to.
rem
rem Leave this window open while you work -- it shows the live server log.
rem Close it (or Ctrl+C, then press Y) to stop the app, or run
rem stop_routing101.bat 8000 from elsewhere.

setlocal
set "PROFILE=768"
set "PORT=8000"
call "%~dp0_run_common.bat"
endlocal
