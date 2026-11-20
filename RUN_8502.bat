@echo off
setlocal
cd /d "%~dp0"

call "env_win\Scripts\activate.bat"
set PYTHONPATH=src

echo ================================================================
echo  Starting the app on port 8502 (8501 is held by an older window).
echo  Your browser should open automatically at:
echo.
echo      http://localhost:8502
echo.
echo  Keep this window open while you use the app.
echo  Press Ctrl+C in this window to stop it when you're done.
echo ================================================================
echo.

streamlit run src\streamlit_app.py --server.port 8502

pause
