@echo off
:: Upload a finished reel to YouTube from the repo root.
:: Usage:
::   run_uploader.bat --file apps\shared\data\exports\reel.mp4 --title "NHL 25 Highlights"
::   run_uploader.bat --file reel.mp4 --title "My Game" --privacy public

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

if not defined APP_DIR set "APP_DIR=%ROOT%"

"%ROOT%\.venv\Scripts\python.exe" "%ROOT%\apps\uploader\upload_youtube.py" %*
