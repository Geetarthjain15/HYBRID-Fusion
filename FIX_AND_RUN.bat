@echo off
setlocal
cd /d "%~dp0"

echo ================================================================
echo   Brain Tumor Detection AI - Fix transformers version and Run
echo ================================================================
echo.
echo The freshly installed "transformers" library (v5) renamed the
echo internal ViT layer names, so it can't load your older .pth
echo checkpoint. Reinstalling a compatible 4.x version...
echo.

call "env_win\Scripts\activate.bat"

pip install "transformers>=4.40,<5" --upgrade
if errorlevel 1 (
    echo.
    echo ERROR: Could not install a compatible transformers version.
    pause
    exit /b 1
)

set PYTHONPATH=src

echo.
echo ================================================================
echo  Starting the app. Your browser should open automatically.
echo  (If port 8501 is already in use by another window, Streamlit
echo   will automatically pick the next free port, e.g. 8502 -
echo   watch this window for the exact "Local URL" it prints.)
echo.
echo  Keep this window open while you use the app.
echo  Press Ctrl+C in this window to stop it when you're done.
echo ================================================================
echo.

streamlit run src\streamlit_app.py

pause
