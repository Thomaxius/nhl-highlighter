@echo off
:: Run the reel builder pipeline from the repo root.
:: Usage:
::   run_reel_builder.bat --file apps\shared\data\mygame.mp4
::   run_reel_builder.bat --folder apps\shared\data\raw
::   run_reel_builder.bat --file mygame.mp4 --output apps\shared\data\exports\reel.mp4

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

"%ROOT%\.venv\Scripts\python.exe" "%ROOT%\apps\reel_builder\pipeline.py" %*
