@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%PYTHON%" goto use_venv

echo [run_gui] .venv not found. Falling back to py -3.11.
py -3.11 "%~dp0gui.py" %*
exit /b %ERRORLEVEL%

:use_venv
"%PYTHON%" "%~dp0gui.py" %*
exit /b %ERRORLEVEL%
