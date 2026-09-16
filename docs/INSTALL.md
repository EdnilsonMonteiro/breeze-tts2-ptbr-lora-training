# Instalação

## Requisitos

| Item | Mínimo |
|---|---|
| SO | Windows 10/11 ou Linux |
| Python | 3.10 – 3.12 |
| GPU | NVIDIA CUDA (treino em 16 GB; inferência ~8 GB) |
| Disco | ~60 GB (corpus + tokens), mais o modelo base (~8 GB) |

## Passos

```bash
git clone https://github.com/EdnilsonMonteiro/breeze-tts2-ptbr-lora-training.git
cd breeze-tts2-ptbr-lora-training

python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux:
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt -r requirements-ptbr.txt
```

- `requirements.txt` é o do **engine** (torch, qwen-tts, transformers, ...).
- `requirements-ptbr.txt` adiciona o que o pipeline usa (peft, pyarrow, scipy,
  scikit-learn, librosa, num2words, pyannote.audio, silero-vad, speechbrain,
  faster-whisper, tensorboard, ...).

## Checkpoint base

O modelo base **não** vem no repositório. Baixe `BreezeBlue/Breeze-TTS-2` do
Hugging Face para `<PTBR_ARTIFACTS>/models/Breeze-TTS-2`:

```bash
python -c "from huggingface_hub import snapshot_download as s; \
s('BreezeBlue/Breeze-TTS-2', local_dir=r'<PTBR_ARTIFACTS>/models/Breeze-TTS-2')"
```

Se o repo for *gated*, aceite os termos e defina `HF_TOKEN` (ver
[CONFIGURATION.md](CONFIGURATION.md)).

## Validando a instalação

```bash
# compila o pacote (rápido, sem GPU)
python -m compileall -q ptbr_lora

# introspecção do modelo base (Fase A)
python ptbr_lora/tools/analyze_model.py
```

## Próximo passo

Configure os caminhos em [CONFIGURATION.md](CONFIGURATION.md) e prepare o dataset
em [DATASETS.md](DATASETS.md).
