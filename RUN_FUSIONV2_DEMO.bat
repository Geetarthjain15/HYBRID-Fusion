@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    set "PYEXE=python"
) else (
    set "PYEXE=py -3"
)

if not exist "env_win\Scripts\python.exe" (
    echo Creating the Python environment...
    %PYEXE% -m venv env_win || exit /b 1
)

echo Installing required packages...
"env_win\Scripts\python.exe" -m pip install --upgrade pip || exit /b 1
"env_win\Scripts\python.exe" -m pip install -r requirements.txt || exit /b 1

echo Downloading and verifying FusionV2 checkpoints...
"env_win\Scripts\python.exe" scripts\download_checkpoints.py || exit /b 1

echo Starting FusionV2 at http://localhost:8503
"env_win\Scripts\python.exe" -m streamlit run src\publication\streamlit_fusion_v2.py --server.port 8503
