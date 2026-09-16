"""paths.py — resolucao unica de caminhos do projeto, dirigida por ambiente.

Regra: o CODIGO vive neste repo (o fork do engine + `ptbr_lora/`), mas os
ARTEFATOS (datasets/, training/, models/) ficam FORA do repositorio git.

Configure a raiz dos artefatos de uma das formas:
  - variavel de ambiente  PTBR_ARTIFACTS=C:\\IA\\Breeze-tts
  - arquivo .env na raiz do repo (mesmo formato KEY=VALUE; ver .env.example)

Variaveis aceitas (todas opcionais; ha default repo-local):
  PTBR_ARTIFACTS       raiz de datasets/, training/, models/
  BREEZE_TTS_REPO      onde esta o engine (default: a raiz deste fork)
  BREEZE_CKPT          pasta do checkpoint base (default: <ARTIFACTS>/models/Breeze-TTS-2)
  BREEZE_DATASETS_DIR  pasta de datasets (default: <ARTIFACTS>/datasets)
  BREEZE_TRAINING_DIR  pasta de training (default: <ARTIFACTS>/training)
  BREEZE_DATASET_DIR   corpus legado unico (default: <datasets>/TTS-Portuguese-Corpus)
  BREEZE_PY            interpretador Python (default: sys.executable)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ptbr_lora/core/paths.py -> parents[2] = raiz do repo (engine + ptbr_lora)
REPO = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    """Carrega KEY=VALUE de <repo>/.env sem depender de python-dotenv."""
    env = REPO / ".env"
    if not env.is_file():
        return
    for raw in env.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv()


def _p(env: str, default: Path) -> Path:
    val = os.environ.get(env)
    return Path(val).expanduser().resolve() if val else default


ARTIFACTS = _p("PTBR_ARTIFACTS", REPO / "artifacts")

# O engine fica na raiz deste fork por padrao; permite apontar para outro clone.
BREEZE_REPO = _p("BREEZE_TTS_REPO", REPO)

CKPT = _p("BREEZE_CKPT", ARTIFACTS / "models" / "Breeze-TTS-2")
DATASETS_ROOT = _p("BREEZE_DATASETS_DIR", ARTIFACTS / "datasets")
TRAINING = _p("BREEZE_TRAINING_DIR", ARTIFACTS / "training")
TOKENS_DIR = TRAINING / "tokens"
WAVS24_DIR = TRAINING / "wavs24"
GOLD_DIR = TRAINING / "gold_samples"

DATASET = _p("BREEZE_DATASET_DIR", DATASETS_ROOT / "TTS-Portuguese-Corpus")
TEXTS_CSV = DATASET / "texts.csv"
CORPORA_JSON = DATASETS_ROOT / "corpora.json"

# Saidas do pipeline de scraping de podcast (codigo vive em ptbr_lora/scraping).
SCRAPING = ARTIFACTS / "dataScrapping"
SCRAPING_WORK = SCRAPING / "work"

# Interpretador para subprocessos (auto_train); default = o proprio Python atual.
BREEZE_PY = os.environ.get("BREEZE_PY") or sys.executable
