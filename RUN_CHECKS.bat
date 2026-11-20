@echo off
setlocal
cd /d "%~dp0"
call "env_win\Scripts\activate.bat"
set PYTHONPATH=src

echo ================================================ > checks_log.txt
echo  PART 1  Pre-training sanity checks             >> checks_log.txt
echo ================================================ >> checks_log.txt
python src\sanity_checks.py >> checks_log.txt 2>&1
set SANITY=%ERRORLEVEL%

echo. >> checks_log.txt
echo ================================================ >> checks_log.txt
echo  PART 2  Duplicate / leakage audit (report only) >> checks_log.txt
echo ================================================ >> checks_log.txt
python src\dedup_audit.py >> checks_log.txt 2>&1

echo. >> checks_log.txt
echo ALL_DONE >> checks_log.txt

echo.
echo ================================================
if "%SANITY%"=="0" (
  echo  Sanity checks PASSED - safe to train.
) else (
  echo  Sanity checks FAILED - see checks_log.txt.
)
echo  Full log:  checks_log.txt
echo  Audit:     dedup_report.json
echo ================================================
echo.
pause
