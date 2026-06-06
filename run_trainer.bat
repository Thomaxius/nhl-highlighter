@echo off
:: Run the VideoMAE fine-tuning trainer from the repo root.
:: Usage:
::   run_trainer.bat
::   run_trainer.bat --data_dir apps\shared\data\labeled --epochs 20

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

"%ROOT%\.venv\Scripts\python.exe" -m apps.trainer.src.training.trainer %*
