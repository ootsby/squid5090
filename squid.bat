@echo off
setlocal
rem Usage: squid.bat [port]   (default 8000). Stop with Ctrl+C.
set "PORT=8000"
if not "%~1"=="" set "PORT=%~1"
set "MODEL=F:\squid\model"
if not exist "%MODEL%\model-00001-of-00001.safetensors" (
  echo Model not found in %MODEL%
  pause
  exit /b 1
)
wslc rm -f squid >nul 2>nul
echo Starting squid5090 on http://localhost:%PORT%/v1  (loading takes about a minute)
wslc run --rm -it --name squid --gpus all --ulimit memlock=-1 -p 127.0.0.1:%PORT%:8000 -v "%MODEL%:/model:ro" squid5090:mix512 ^
  serve /model --host 0.0.0.0 --port 8000 --mode nvfp4 --kv q4 --ctx 32768 --mtp 7 --embed-host
if errorlevel 1 pause
