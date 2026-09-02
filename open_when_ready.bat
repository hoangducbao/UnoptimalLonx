@echo off
rem open_when_ready.bat [port] -- polls the app until it responds, then opens
rem it in the default browser. Launched minimized/in the background by
rem run_768.bat / run_1152.bat, so the model-loading wait doesn't block that
rem window. The port defaults to 8000 (the 768-dim profile); run_1152.bat
rem passes 8001 for the 1152-dim one.

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8000"

:loop
curl -s -o nul -f http://localhost:%PORT%/app/
if errorlevel 1 (
    timeout /t 2 >nul
    goto loop
)
start "" http://localhost:%PORT%/app/
exit
