@echo off
:: =============================================================================
:: PrimateScope AI — One-Click Setup Script (Windows)
:: =============================================================================
:: Run this ONCE before first use. Python 3.11 or 3.12 required (NOT 3.14).

setlocal enabledelayedexpansion

echo.
echo ================================================================
echo   PrimateScope AI - Field Intelligence System (Production v1.0)
echo   Setting up your environment...
echo ================================================================
echo.

:: --- Step 1: Python check ---
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found.
    echo Install Python 3.11 or 3.12 from: https://www.python.org/downloads/
    echo NOTE: During installation, check "Add Python to PATH".
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set PYMAJOR=%%a
    set PYMINOR=%%b
)

if %PYMINOR% GEQ 14 (
    echo [ERROR] Python %PYTHON_VERSION% is NOT supported.
    echo SpeciesNet/MegaDetector may not install on Python 3.14.
    echo Install Python 3.11 or 3.12 from: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo    Found Python %PYTHON_VERSION%

:: --- Step 2: Create virtual environment ---
echo.
echo [1/5] Creating virtual environment...
if exist "venv" (
    echo    Skipping - venv already exists.
) else (
    python -m venv venv
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
)
echo    Done.

:: --- Step 3: Upgrade pip ---
echo.
echo [2/5] Upgrading pip...
call venv\Scripts\python.exe -m pip install --upgrade pip -q

:: --- Step 4: Install core dependencies ---
echo.
echo [3/5] Installing core dependencies (~2 min)...
call venv\Scripts\pip.exe install -q -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Core dependency installation failed.
    pause
    exit /b 1
)
echo    Done.

:: --- Step 5: Try SpeciesNet + MegaDetector ---
echo.
echo [4/5] Installing SpeciesNet + MegaDetector (optional)...
call venv\Scripts\pip.exe install speciesnet megadetector --use-pep517 -q 2>nul
if %ERRORLEVEL% neq 0 (
    echo    [WARN] SpeciesNet/MegaDetector install failed.
    echo           App runs in Demo mode. Real inference requires these.
) else (
    echo    Done.
)

:: --- Step 6: YOLOv8n model ---
echo.
echo [5/5] Downloading YOLOv8n model (~6 MB)...
if exist "yolov8n.pt" (
    echo    Model already present.
) else (
    call venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" >nul 2>&1
    if exist "%USERPROFILE%\Ultralytics\yolov8n.pt" copy "%USERPROFILE%\Ultralytics\yolov8n.pt" "yolov8n.pt" >nul
)

:: --- Step 7: Environment check ---
echo.
echo Running environment check...
call venv\Scripts\python.exe scripts\check_environment.py

:: --- Done ---
echo.
echo ================================================================
echo   SETUP COMPLETE!
echo.
echo   Next step - launch the app:
echo   Double-click: LAUNCH.bat
echo.
echo   The app opens at http://localhost:8501
echo   Default mode: Demo Simulation
echo   For real inference: switch to Real Inference mode in the sidebar
echo ================================================================
echo.
pause
