@echo off
setlocal
cd /d "%~dp0"

REM ---------------------------------------------------------------------
REM  Build a SEPARATE CUDA environment (env_cuda).
REM
REM  env_win is NOT touched. If this fails for any reason, the existing
REM  working environment is unaffected and you lose nothing. Only switch
REM  the .bat runners over to env_cuda once the verification below passes.
REM
REM  Why separate: the transformers v5 incident showed that upgrading in
REM  place can break a working pipeline. A ~2.5 GB download is cheaper
REM  than a broken environment.
REM ---------------------------------------------------------------------

echo ================================================ > gpu_setup_log.txt
echo  Step 0  Driver / CUDA visible to Windows       >> gpu_setup_log.txt
echo ================================================ >> gpu_setup_log.txt
nvidia-smi >> gpu_setup_log.txt 2>&1
if errorlevel 1 (
  echo.
  echo  nvidia-smi failed - no NVIDIA driver found.
  echo  Nothing was installed. See gpu_setup_log.txt.
  echo.
  pause
  exit /b 1
)

echo. >> gpu_setup_log.txt
echo ================================================ >> gpu_setup_log.txt
echo  Step 1  Create env_cuda                        >> gpu_setup_log.txt
echo ================================================ >> gpu_setup_log.txt
if not exist "env_cuda" (
  python -m venv env_cuda >> gpu_setup_log.txt 2>&1
)
call "env_cuda\Scripts\activate.bat"
python -m pip install --upgrade pip >> gpu_setup_log.txt 2>&1

echo. >> gpu_setup_log.txt
echo ================================================ >> gpu_setup_log.txt
echo  Step 2  Install CUDA torch (cu124 wheels)      >> gpu_setup_log.txt
echo ================================================ >> gpu_setup_log.txt
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 >> gpu_setup_log.txt 2>&1

echo. >> gpu_setup_log.txt
echo ================================================ >> gpu_setup_log.txt
echo  Step 3  Install the rest (same pins as before) >> gpu_setup_log.txt
echo ================================================ >> gpu_setup_log.txt
python -m pip install -r requirements.txt >> gpu_setup_log.txt 2>&1

echo. >> gpu_setup_log.txt
echo ================================================ >> gpu_setup_log.txt
echo  Step 4  VERIFY                                 >> gpu_setup_log.txt
echo ================================================ >> gpu_setup_log.txt
python -c "import torch,sys; ok=torch.cuda.is_available(); print('torch',torch.__version__); print('cuda_available',ok); print('device',torch.cuda.get_device_name(0) if ok else 'NONE'); sys.exit(0 if ok else 1)" >> gpu_setup_log.txt 2>&1
set VERIFY=%ERRORLEVEL%

echo. >> gpu_setup_log.txt
echo ALL_DONE >> gpu_setup_log.txt

echo.
echo ================================================
if "%VERIFY%"=="0" (
  echo  SUCCESS - env_cuda sees the GPU.
  echo  env_win is untouched and still works.
  echo  To use the GPU, change "env_win" to "env_cuda"
  echo  in the .bat runner you want to speed up.
) else (
  echo  FAILED - env_cuda cannot see the GPU.
  echo  Nothing else changed; keep using env_win.
  echo  Check gpu_setup_log.txt for the reason.
)
echo  Log: gpu_setup_log.txt
echo ================================================
echo.
pause
