@echo off
title DocuMind AI Launcher
echo ========================================================
echo Starting DocuMind AI...
echo ========================================================
echo.

cd /d "%~dp0"

echo Opening DocuMind AI in your browser...
echo If the browser does not open automatically, open:
echo http://localhost:8501
echo.

python -m streamlit run app.py --browser.gatherUsageStats false

pause
