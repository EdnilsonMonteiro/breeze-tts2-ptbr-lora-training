@echo off
REM Sweep completo: 50 seeds (config padrao) + grade de configs + metricas.
REM Roda em C:\IA\Breeze-tts2-ptbr-lora-training\ptbr_lora\teste_voz
setlocal
cd /d "C:\IA\Breeze-tts2-ptbr-lora-training\ptbr_lora\teste_voz"
set "PY=C:\IA\Breeze-tts\breeze-tts\venv\Scripts\python.exe"
if not exist teste_seeds mkdir teste_seeds
set "LOG=teste_seeds\sweep_console.log"
echo [run] inicio %DATE% %TIME% > "%LOG%"

echo [run] FASE A: 50 seeds (padrao) >> "%LOG%"
"%PY%" seed_sweep.py --seeds 1-50 --out-dir teste_seeds >> "%LOG%" 2>&1

echo [run] FASE B: grade de configs >> "%LOG%"
"%PY%" find_best_config.py --build-grid --out-dir teste_seeds >> "%LOG%" 2>&1
"%PY%" seed_sweep.py --seeds 1-5 --configs teste_seeds\grid.json --out-dir teste_seeds >> "%LOG%" 2>&1

echo [run] FASE C: metricas + ranking >> "%LOG%"
"%PY%" compare_seeds.py --dir teste_seeds --out teste_seeds\metrics.csv >> "%LOG%" 2>&1
"%PY%" find_best_config.py --analyze teste_seeds\metrics.csv --out-dir teste_seeds >> "%LOG%" 2>&1

echo [run] FIM %DATE% %TIME% >> "%LOG%"
endlocal
