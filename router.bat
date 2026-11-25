@echo off
rem Usage: router.bat [--port 8000]   Models and idle time are in router.json. Stop with Ctrl+C.
python "%~dp0router.py" %*
if errorlevel 1 pause
