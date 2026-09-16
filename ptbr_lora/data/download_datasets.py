"""download_datasets.py — baixa os corpora novos via HuggingFace (hf_transfer).

- TAGARELA: split `tts` (clean) via branch refs/convert/parquet, 8 shards intercalados.
- CML-TTS PT: freds0/cml_tts_dataset_portuguese, split train (29 shards).
- CETUC: falabrasil/cetuc, ~24 locutores do split train (WebDataset tar.gz).

Uso: python scripts/download_datasets.py [--only tagarela|cml|cetuc]
Resumivel (snapshot_download pula arquivos ja baixados).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
import paths  # noqa: E402

RAW = paths.DATASETS_ROOT / "_raw"

TAGARELA_SHARDS = [0, 4, 9, 13, 18, 22, 27, 31, 36, 45, 54, 63]
TAGARELA_PATTERNS = [f"default/tts/{i:04d}.parquet" for i in TAGARELA_SHARDS]

CETUC_SPEAKERS = [
    "AdrianaMalta_F049", "Alcione_F018", "Alessandra_F045", "AnaVarela_F042",
    "Andrea_F003", "Carla_F035", "ClaudiaMoraes_F023", "Cristiane_F007",
    "Flavia_F047", "Geruza_F006", "Ieda_F014", "Juliana_F028", "Madel_F002",
    "Mariana_F024", "Milena_F044", "Paula_F026", "PriscilaTerra_F004",
    "Regina_F013", "SilvanaFerreira_F012", "TatianaRuback_F038",
    "Aislam_M001", "Diego_M026", "Elson_M007", "FabioCorrea_M032",
    "Gilberto_M018", "IvanMariano_M008", "Jair_M021", "Joel_M017",
    "JulioFaustino_M005", "Marcio_M011", "Oswaldo_M012", "Paulinho_M000",
    "RenatoPeres_M010", "Rogerio_M035", "Tulio_M027", "Walace_M004",
]
CETUC_PATTERNS = [f"data/train/{s}/{s}.tar.gz" for s in CETUC_SPEAKERS]


def run(name: str, repo_id: str, patterns: list[str], local_dir: Path, revision=None):
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[dl] ===== {name}: {repo_id} ({len(patterns)} arquivos) =====", flush=True)
    t0 = time.time()
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=patterns,
        local_dir=str(local_dir),
        max_workers=8,
    )
    print(f"[dl] {name} OK em {(time.time()-t0)/60:.1f} min -> {local_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["tagarela", "cml", "cetuc"], default=None)
    args = ap.parse_args()

    jobs = {
        "tagarela": ("freds0/TAGARELA", TAGARELA_PATTERNS, RAW / "tagarela",
                     "refs/convert/parquet"),
        "cml": ("freds0/cml_tts_dataset_portuguese", ["data/train-*"], RAW / "cml_pt", None),
        "cetuc": ("falabrasil/cetuc", CETUC_PATTERNS, RAW / "cetuc", None),
    }
    order = [args.only] if args.only else ["tagarela", "cml", "cetuc"]
    for key in order:
        repo_id, patterns, local_dir, rev = jobs[key]
        run(key, repo_id, patterns, local_dir, rev)
    print("\n[dl] TODOS OS DOWNLOADS CONCLUIDOS", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
