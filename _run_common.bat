@echo off
rem _run_common.bat -- shared launch sequence behind run_768.bat and
rem run_1152.bat. Not meant to be run directly: the entry-point scripts set
rem PROFILE and PORT, then `call` this.
rem
rem Everything here is profile-independent on purpose -- one Docker check,
rem one Elasticsearch container, one uvicorn invocation -- so a fix to the
rem bootstrap lands for both profiles at once instead of being copy-pasted
rem into two launchers and drifting.

if "%PROFILE%"=="" (
    echo _run_common.bat is not meant to be run directly.
    echo Use run_768.bat or run_1152.bat instead.
    exit /b 1
)

cd /d "%~dp0"
set "R101_EMBED=%PROFILE%"
title Routing101 %PROFILE%d

echo === Routing101 launcher ===
echo Embedding profile: %PROFILE%-dim   Port: %PORT%
echo.

rem Docker Desktop's install path is hardcoded below. If it isn't there we
rem can't auto-start it, so say so plainly and wait for the user rather than
rem spinning silently forever on a machine where the path differs.
echo Checking Docker...
docker info >nul 2>&1
if not errorlevel 1 goto dockerup
if exist "C:\Program Files\Docker\Docker\Docker Desktop.exe" (
    echo Docker isn't running -- starting Docker Desktop, this can take a minute or two...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
) else (
    echo Docker isn't running, and Docker Desktop was not found at:
    echo     C:\Program Files\Docker\Docker\Docker Desktop.exe
    echo Start Docker yourself -- this window will keep waiting until it's up.
    echo ^(Or edit that path in _run_common.bat if yours is installed elsewhere.^)
)
:waitdocker
timeout /t 3 >nul
docker info >nul 2>&1
if errorlevel 1 goto waitdocker
echo Docker is up.
:dockerup

rem One "es" container serves both profiles -- the Elasticsearch indices are
rem text-only and dimension-independent, so the second profile to start just
rem finds them already there.
echo Starting Elasticsearch container...
docker start es >nul 2>&1
if errorlevel 1 (
    echo No existing "es" container found -- creating one...
    docker run -d --name es -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" -e "xpack.ml.enabled=false" -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" -v es-data:/usr/share/elasticsearch/data docker.elastic.co/elasticsearch/elasticsearch:8.15.0
)

echo Waiting for Elasticsearch on :9200...
:waites
curl -s -o nul -f http://localhost:9200 >nul 2>&1
if errorlevel 1 (
    timeout /t 2 >nul
    goto waites
)
echo Elasticsearch is up.
echo.

start "" /min cmd /c "%~dp0open_when_ready.bat" %PORT%

echo Starting Routing101 %PROFILE%d on :%PORT% (uvicorn --reload) -- backend/*.py edits auto-restart it.
echo The frontend will open in your browser automatically once it's ready.
echo A cold start on a profile with no index/ yet spends several minutes building one.
echo.
python -m uvicorn backend.main:app --reload --port %PORT%
