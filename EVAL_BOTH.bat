@echo off
setlocal
cd /d "%~dp0"
call "env_win\Scripts\activate.bat"
set PYTHONPATH=src

echo Evaluating Original ViT on the held-out test set... > eval_output.txt
python src\eval.py --model vit --data_dir dataset\Testing >> eval_output.txt 2>&1

echo. >> eval_output.txt
echo Evaluating Hybrid CNN+ViT on the held-out test set... >> eval_output.txt
python src\eval.py --model hybrid --data_dir dataset\Testing >> eval_output.txt 2>&1

echo DONE >> eval_output.txt
echo.
echo Finished. See eval_output.txt in this folder.
pause
