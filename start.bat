@echo off
setlocal enabledelayedexpansion
title AuraCoreComfyUI

echo.
echo  ==========================================
echo    AuraCoreComfyUI - Launcher
echo  ==========================================
echo.

:: All paths are relative to the folder this .bat lives in
set "ROOT=%~dp0"
set "VENV=%ROOT%venv"
set "VENV_PYTHON=%VENV%\Scripts\python.exe"
set "VENV_PIP=%VENV%\Scripts\pip.exe"
set "VENV_ACTIVATE=%VENV%\Scripts\activate.bat"

:: ── Check system Python is available ───────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found in PATH.
    echo  Please install Python 3.10+ and make sure it is added to PATH.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PY_VER=%%i
echo  [OK] Found %PY_VER%

:: ── Create venv if it doesn't exist ────────────────────
if not exist "%VENV_PYTHON%" (
    echo.
    echo  [INFO] No venv found. Creating one at: %VENV%
    python -m venv "%VENV%"
    if errorlevel 1 (
        echo  [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
    echo  [OK] venv created.
) else (
    echo  [OK] venv found at: %VENV%
)

:: ── Check requirements.txt exists ──────────────────────
if not exist "%ROOT%requirements.txt" (
    echo  [ERROR] requirements.txt not found.
    pause
    exit /b 1
)

:: ── Install / update requirements into the venv ────────
echo.
echo  Checking requirements...
echo.

"%VENV_PIP%" install -r "%ROOT%requirements.txt" --quiet --disable-pip-version-check

if errorlevel 1 (
    echo.
    echo  [ERROR] Failed to install one or more requirements.
    echo  Check your internet connection or try deleting the venv\ folder and restarting.
    pause
    exit /b 1
)
echo  [OK] Requirements satisfied.

:: ── Check app.py exists ────────────────────────────────
if not exist "%ROOT%app.py" (
    echo  [ERROR] app.py not found.
    pause
    exit /b 1
)

:: ── Launch app using the venv Python ───────────────────
echo.
echo  Starting AuraCoreComfyUI...
echo  Open http://localhost:7860 in your browser.
echo  (It will open automatically)
echo.
echo  Press Ctrl+C to stop the app.
echo.

cd /d "%ROOT%"
"%VENV_PYTHON%" app.py

:: ── Pause on exit so errors are visible ────────────────
echo.
echo  App stopped.
pause
