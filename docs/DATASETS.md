# Dados: download, ingestão e preparação

O pipeline transforma áudio + texto em **tokens RVQ pré-extraídos** e em
**splits** prontos para o treino. Os dados ficam em `<PTBR_ARTIFACTS>/datasets/`
e `<PTBR_ARTIFACTS>/training/`.

## 1. Download de corpora públicos (Hugging Face)

```bash
python ptbr_lora/data/download_datasets.py            # todos
python ptbr_lora/data/download_datasets.py --only tagarela
python ptbr_lora/data/download_datasets.py --only cml
python ptbr_lora/data/download_datasets.py --only cetuc
```

Baixa para `datasets/_raw/<corpus>/` via `snapshot_download` (resumível). Para
repos *gated*, defina `HF_TOKEN`.

## 2. Ingestão para o formato do projeto

```bash
python ptbr_lora/data/build_corpus.py --corpus tagarela --hours 120
python ptbr_lora/data/build_corpus.py --corpus cml_pt   --hours 35
python ptbr_lora/data/build_corpus.py --corpus cetuc    --hours 30
```

Gera `datasets/<corpus>/wavs/<idx>.wav` + `texts.csv` + `speakers.jsonl` +
`selection.jsonl` (manifesto). Filtros: duração 4,0–10,2 s, densidade textual
4–30 chars/s, números → extenso (pt-BR), NFC. Resumível (pula itens já no
`selection.jsonl`).

### Locutores do TAGARELA (clustering ECAPA)

TAGARELA não traz rótulo de locutor por linha. O script agrupa por voz dentro de
cada show via embeddings ECAPA + clustering aglomerativo e produz
`speakers.jsonl` e `ref_map.jsonl` (referência do **mesmo** locutor):

```bash
python ptbr_lora/data/tagarela_speakers.py
```

### Podcast próprio (opcional)

`ptbr_lora/scraping/` converte episódios brutos (48 kHz estéreo) em chunks 24 kHz
mono + transcrição (`00_preprocess` → `01_diarize` → `02_slice` → `03_transcribe`
→ `04_qc_audit`). Requer `XAI_TRANSCRIBE_KEY` e `HF_TOKEN` no `.env`. Detalhes no
cabeçalho de cada script e no `scraping/README.md`.

## 3. Preparação para o treino

Dois comandos (o `process` é resumível — pula `.npz` existentes):

```bash
python ptbr_lora/core/prepare_dataset.py process [--limit N] [--device cuda]
python ptbr_lora/core/prepare_dataset.py finalize [--gold N] [--parity-device cuda]
```

- **process**: lê os `texts.csv` dos corpora ativos, reamostra para 24 kHz,
  codifica o áudio uma única vez (`Qwen3TTSTokenizer`) e salva
  `training/tokens/<idx>.npz`; gera `wavs24/`.
- **finalize**: split estratificado **90/5/5 por corpus** (`splits_{train,val,test}.txt`),
  `dataset_meta.jsonl`, `manifest.csv`, `summary.json`, amostras *gold* e a
  verificação de paridade com o template oficial.

## Formato do `texts.csv`

Uma linha por item, no formato `wavs/<idx>.wav==texto`:

```
wavs/sample-1.wav==Ele trabalha no escritório desde o ano passado.
wavs/podcast_000123.wav==A previsão do tempo indica pancadas de chuva à tarde.
```

O parser aceita esse formato direto; `speakers.jsonl` (opcional) acrescenta
`{"idx","speaker"}` e `ref_map.jsonl` `{"idx","ref_idx","cos"}`.

## Próximo passo

[`TRAINING.md`](TRAINING.md).
