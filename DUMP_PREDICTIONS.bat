@echo off
setlocal
cd /d "%~dp0"
call "env_win\Scripts\activate.bat"
set PYTHONPATH=src

echo ===== 1/4  Hybrid CNN+ViT on dataset\Testing ===== > metrics_run_log.txt
python src\dump_predictions.py --model hybrid --data_dir dataset\Testing --out_csv preds_hybrid_test.csv >> metrics_run_log.txt 2>&1

echo. >> metrics_run_log.txt
echo ===== 2/4  Original ViT on dataset\Testing ===== >> metrics_run_log.txt
python src\dump_predictions.py --model vit --data_dir dataset\Testing --out_csv preds_vit_test.csv >> metrics_run_log.txt 2>&1

echo. >> metrics_run_log.txt
echo ===== 3/4  Hybrid CNN+ViT on dataset_ood\Testing ===== >> metrics_run_log.txt
python src\dump_predictions.py --model hybrid --data_dir dataset_ood\Testing --out_csv preds_hybrid_ood.csv >> metrics_run_log.txt 2>&1

echo. >> metrics_run_log.txt
echo ===== 4/4  Original ViT on dataset_ood\Testing ===== >> metrics_run_log.txt
python src\dump_predictions.py --model vit --data_dir dataset_ood\Testing --out_csv preds_vit_ood.csv >> metrics_run_log.txt 2>&1

echo. >> metrics_run_log.txt
echo ALL_DONE >> metrics_run_log.txt

echo.
echo ================================================
echo  Finished. Four CSV files written in this folder:
echo    preds_hybrid_test.csv
echo    preds_vit_test.csv
echo    preds_hybrid_ood.csv
echo    preds_vit_ood.csv
echo  Full log: metrics_run_log.txt
echo ================================================
echo.
pause
