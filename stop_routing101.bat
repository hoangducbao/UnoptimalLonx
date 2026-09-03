@echo off
rem stop_routing101.bat [port] -- stops the backend started by run_768.bat,
rem run_1152.bat or run_1536.bat (finds what's listening on that port and its
rem uvicorn --reload parent process, kills both) without touching the
rem Elasticsearch container -- ES stays up so the next run_routing101.bat
rem launch skips its slow cold-start. Run `docker stop es` yourself if you
rem want that stopped too.
rem
rem The port defaults to 8000, the 768-dim profile (run_768.bat). Pass 8001
rem for the 1152-dim one (run_1152.bat) or 8002 for the 1536-dim one
rem (run_1536.bat); they are separate processes, so stopping one leaves the
rem others running.

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8000"

echo === Stopping Routing101 on :%PORT% ===
powershell -NoProfile -Command ^
    "$p = Get-NetTCPConnection -LocalPort %PORT% -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty OwningProcess;" ^
    "if (-not $p) { Write-Output 'No Routing101 backend found listening on port %PORT%.'; exit }" ^
    "$parent = (Get-CimInstance Win32_Process -Filter \"ProcessId=$p\").ParentProcessId;" ^
    "Stop-Process -Id $p -Force -ErrorAction SilentlyContinue;" ^
    "if ($parent) { Stop-Process -Id $parent -Force -ErrorAction SilentlyContinue };" ^
    "Write-Output 'Backend stopped.'"

echo.
echo Elasticsearch container "es" is left running (fast next launch).
echo To stop it too:   docker stop es
pause
