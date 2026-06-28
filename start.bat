@echo off
setlocal enabledelayedexpansion
title AuraCoreComfyUI

echo.
echo  ==========================================
echo    AuraCoreComfyUI - Launcher
echo  ==========================================
echo.

:: ── Check Python is available ──────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found in PATH.
    echo  Please install Python 3.10+ and make sure it is added to PATH.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PY_VER=%%i
echo  [OK] Found %PY_VER%

:: ── Check pip is available ─────────────────────────────
pip --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] pip not found. Run: python -m ensurepip
    pause
    exit /b 1
)
echo  [OK] pip found

:: ── Check requirements.txt exists ──────────────────────
if not exist "%~dp0requirements.txt" (
    echo  [ERROR] requirements.txt not found next to this .bat file.
    pause
    exit /b 1
)

:: ── Check & install requirements ───────────────────────
echo.
echo  Checking requirements...
echo.

pip install -r "%~dp0requirements.txt" --quiet --disable-pip-version-check

if errorlevel 1 (
    echo.
    echo  [ERROR] Failed to install one or more requirements.
    echo  Try running this bat as Administrator or check your internet connection.
    pause
    exit /b 1
)

echo  [OK] Requirements satisfied.

:: ── Check app.py exists ────────────────────────────────
if not exist "%~dp0app.py" (
    echo  [ERROR] app.py not found next to this .bat file.
    pause
    exit /b 1
)

:: ── Launch the app ──────────────────────────────────────
echo.
echo  Starting AuraCoreComfyUI...
echo  Open http://localhost:7860 in your browser.
echo  (It will open automatically)
echo.
echo  Press Ctrl+C to stop the app.
echo.

cd /d "%~dp0"
python app.py

:: ── If app exits, pause so you can see any error ───────
echo.
echo  App stopped.
pause
