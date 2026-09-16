# Configuração e artefatos

O código vive no repositório; os **artefatos** (`datasets/`, `training/`,
`models/`) ficam **fora do git**. Toda a resolução de caminho está em
`ptbr_lora/core/paths.py` e é dirigida por variáveis de ambiente (ou por um
arquivo `.env` na raiz do repo).

## Configuração mínima

Copie `.env.example` para `.env` e edite:

```ini
PTBR_ARTIFACTS=C:\IA\Breeze-tts
```

`PTBR_ARTIFACTS` é a raiz que contém `datasets/`, `training/` e `models/`.

## Variáveis

| Variável | Default | Descrição |
|---|---|---|
| `PTBR_ARTIFACTS` | `./artifacts` | raiz dos artefatos |
| `BREEZE_TTS_REPO` | raiz do fork | onde está o engine (engine já vive na raiz) |
| `BREEZE_CKPT` | `<ARTIFACTS>/models/Breeze-TTS-2` | checkpoint base |
| `BREEZE_DATASETS_DIR` | `<ARTIFACTS>/datasets` | corpora e `corpora.json` |
| `BREEZE_TRAINING_DIR` | `<ARTIFACTS>/training` | `tokens/`, `runs/`, splits, manifestos |
| `BREEZE_DATASET_DIR` | `<datasets>/TTS-Portuguese-Corpus` | corpus legado único |
| `BREEZE_PY` | `sys.executable` | interpretador usado por `auto_train.py` |

## Árvore de artefatos

```
<PTBR_ARTIFACTS>/
├─ models/Breeze-TTS-2/          # checkpoint base (download oficial)
├─ datasets/
│  ├─ corpora.json               # registro de corpora ativos (lista)
│  ├─ TTS-Portuguese-Corpus/     # wavs/ + texts.csv (+ speakers.jsonl/ref_map.jsonl)
│  ├─ tagarela/  cml_pt/  cetuc/ podcast/
│  └─ _raw/                      # downloads brutos (HF)
└─ training/
   ├─ tokens/<idx>.npz           # codecs RVQ pré-extraídos
   ├─ wavs24/                    # áudio 24 kHz normalizado
   ├─ gold_samples/              # amostras de verificação
   ├─ dataset_meta.jsonl         # manifesto por item (idx, text, dur, corpus, speaker)
   ├─ manifest.csv               # manifesto tabular
   ├─ summary.json               # resumo do finalize
   ├─ splits_{train,val,test}.txt
   └─ runs/<run>/
      ├─ config.json  log.csv  trainable_modules.txt
      ├─ checkpoints/<tag>/      # adapter LoRA (adapter_config.json + .safetensors)
      ├─ samples/                # WAVs de validação + wer.json
      └─ tb/                     # TensorBoard
```

## Registrar corpora

`datasets/corpora.json` é uma lista de corpora ativos:

```json
[
  {"name": "tata",     "root": "TTS-Portuguese-Corpus", "csv": "texts.csv", "enabled": true},
  {"name": "podcast",  "root": "podcast",               "csv": "texts.csv", "enabled": true},
  {"name": "tagarela", "root": "tagarela",              "csv": "texts.csv", "speakers": "speakers.jsonl"},
  {"name": "cml_pt",   "root": "cml_pt",                "csv": "texts.csv"},
  {"name": "cetuc",    "root": "cetuc",                 "csv": "texts.csv"}
]
```

Sem esse arquivo, o pipeline cai no par legado (`tata` + `podcast`) dirigido por
`BREEZE_DATASET_DIR`.
