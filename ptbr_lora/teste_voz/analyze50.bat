@echo off
cd /d "C:\IA\Breeze-tts2-ptbr-lora-training\ptbr_lora\teste_voz"
"C:\IA\Breeze-tts\breeze-tts\venv\Scripts\python.exe" compare_seeds.py --dir teste_seeds\T0.9_k50_p1_cfg1 --out teste_seeds\metrics_seeds50.csv --no-plots > teste_seeds\analysis50.log 2>&1
