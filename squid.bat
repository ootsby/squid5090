@echo off
setlocal
rem Usage: squid.bat [port] [mix or nvfp4] [abl]   (default 8000 mix, no ablation). Stop with Ctrl+C.
set "PORT=8000"
if not "%~1"=="" set "PORT=%~1"
set "MODE=mix"
if not "%~2"=="" set "MODE=%~2"
set "ABL="
if /i "%~3"=="abl" set "ABL=--ablation /model/inkling-2-mini-abliterated.safetensors"
set "MODEL=F:\squid\model"
if not exist "%MODEL%\model-00001-of-00001.safetensors" (
  echo Model not found in %MODEL%
  pause
  exit /b 1
)
wslc rm -f squid >nul 2>nul
echo Starting squid5090 on http://localhost:%PORT%/v1  (loading takes about a minute)
wslc run --rm -it --name squid --gpus all --ulimit memlock=-1 -p 127.0.0.1:%PORT%:8000 -v "%MODEL%:/model:ro" squid5090:mixd512 ^
  serve /model --host 0.0.0.0 --port 8000 --mode %MODE% --kv q4 --ctx 32768 --mtp 7 --embed-host %ABL%
if errorlevel 1 pause
