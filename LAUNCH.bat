@echo off
:: =============================================================================
:: PrimateScope AI — Launch Script (Windows)
:: =============================================================================
if not exist "venv" (
    echo [ERROR] venv not found. Run SETUP.bat first.
    pause
    exit /b 1
)

echo.
echo ================================================================
echo   Starting PrimateScope AI...
echo ================================================================
echo.
echo   Opening: http://localhost:8501
echo   Close this window or press Ctrl+C to stop.
echo.

start http://localhost:8501

venv\Scripts\streamlit.exe run app.py ^
    --server.headless true ^
    --server.port 8501 ^
    --browser.gatherUsageStats false ^
    --theme.base dark ^
    --theme.primaryColor "#2EEAD3" ^
    --theme.backgroundColor "#0A0F0D" ^
    --theme.secondaryBackgroundColor "#141C19" ^
    --theme.textColor "#FFFFFF"
