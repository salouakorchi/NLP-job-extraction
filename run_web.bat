@echo off
setlocal
cd /d "%~dp0"

if not exist ".env" (
  copy ".env.example" ".env" >nul
)

python --version >nul 2>nul
if errorlevel 1 (
  echo Python est introuvable. Installez Python 3.10 ou plus, puis relancez ce fichier.
  pause
  exit /b 1
)

python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Installation des dependances impossible.
  pause
  exit /b 1
)

set WEB_PORT=5000
start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; Start-Process 'http://127.0.0.1:%WEB_PORT%/'"
python web_app.py

pause
