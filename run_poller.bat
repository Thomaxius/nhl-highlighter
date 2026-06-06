@echo off
:: Run the YouTube poller from the repo root.
:: Required env vars:
::   YOUTUBE_CHANNEL_ID
::   OAUTH_CLIENT_SECRETS  (default: config\client_secrets.json)
::   OAUTH_TOKEN_FILE      (default: config\token.json)
::   APP_DIR               (default: current directory)

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

if not defined APP_DIR set "APP_DIR=%ROOT%"
if not defined OAUTH_CLIENT_SECRETS set "OAUTH_CLIENT_SECRETS=%ROOT%\config\client_secrets.json"
if not defined OAUTH_TOKEN_FILE set "OAUTH_TOKEN_FILE=%ROOT%\config\token.json"

"%ROOT%\.venv\Scripts\python.exe" "%ROOT%\apps\poller\youtube_watcher.py" %*
