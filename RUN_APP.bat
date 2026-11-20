@echo off
setlocal
cd /d "%~dp0"

echo ================================================================
echo   Brain Tumor Detection AI - Setup and Run  (Windows)
echo ================================================================
echo.

REM --- Find a Python interpreter ---
set PYEXE=python
where python >nul 2>nul
if errorlevel 1 (
    where py >nul 2>nul
    if errorlevel 1 (
        echo ERROR: Python was not found on PATH.
        echo Install Python 3.10+ from https://www.python.org/downloads/
        echo ^(tick "Add python.exe to PATH" during install^), then re-run this file.
        pause
        exit /b 1
    ) else (
        set PYEXE=py
    )
)

REM --- The "env" folder that shipped with this project was created on a
REM     Mac and will NOT work on Windows. We create a separate, fresh
REM     Windows virtual environment called "env_win" instead. ---
if not exist "env_win\Scripts\activate.bat" (
    echo Creating a Windows virtual environment in env_win ...
    %PYEXE% -m venv env_win
    if errorlevel 1 (
        echo ERROR: Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

call "env_win\Scripts\activate.bat"

echo.
echo Upgrading pip...
python -m pip install --upgrade pip >nul

echo.
echo Installing project dependencies from requirements.txt
echo (first run only - this downloads PyTorch etc. and can take
echo  10-20+ minutes depending on your internet connection)...
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: pip install failed - see the messages above for details.
    pause
    exit /b 1
)

set PYTHONPATH=src

echo.
echo ================================================================
echo  Starting the app. Your browser should open automatically at:
echo.
echo      http://localhost:8501
echo.
echo  Keep this window open while you use the app.
echo  Press Ctrl+C in this window to stop it when you're done.
echo ================================================================
echo.

streamlit run src\streamlit_app.py

pause
