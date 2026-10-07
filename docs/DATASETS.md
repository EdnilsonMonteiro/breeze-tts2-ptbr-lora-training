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

**Identidade dos locutores (obrigatório antes de treinar com podcast):** a diarização dá rótulos por episódio (`SPEAKER_00` de um episódio não tem relação com o de outro) e erra nas bordas e em fala sobreposta. `python ptbr_lora/data/podcast_identity.py` (GPU, ~5–10 min) corrige isso com embeddings de voz:

1. descarta clipes cuja margem invade fala de outro locutor (RTTM) e clipes com troca de voz no meio (1ª × 2ª metade);
2. limpa cada rótulo (tira intrusos; rótulo com 2 vozes vira `#a`/`#b`);
3. liga a **mesma pessoa em episódios diferentes** num id global `podcast:P###` (ligação completa, limiar calibrado em pares sabidamente diferentes do CETUC);
4. episódios que compartilham alguém viram um grupo (`G###`): a mesma pessoa nunca fica em treino e validação.

Saída no padrão dos outros corpora: `datasets/podcast/speakers.jsonl` (`idx`, `speaker`, `group`; descartados = `podcast:99`) e `datasets/podcast/identity/` (`report.html` para ouvir as decisões, `chunks.csv`, `speakers.csv`, `summary.json`). Use `--dry-run` para só gerar o relatório.

## 3. Preparação para o treino

Dois comandos (o `process` é resumível — pula `.npz` existentes):

```bash
python ptbr_lora/core/prepare_dataset.py process [--limit N] [--device cuda]
python ptbr_lora/core/prepare_dataset.py finalize [--gold N] [--parity-device cuda]
```

- **process**: lê os `texts.csv` dos corpora ativos, reamostra para 24 kHz,
  codifica o áudio uma única vez (`Qwen3TTSTokenizer`) e salva
  `training/tokens/<idx>.npz`; gera `wavs24/`.
- **finalize**: split **90/5/5 por corpus e por GRUPO** (`splits_{train,val,test}.txt`; grupo =
  programa/episódio no tagarela e no podcast, locutor nos demais — `core/splits.py`, seed 42, com
  auditoria de vazamento), `manifest.csv`, `summary.json`, amostras *gold* e a verificação de
  paridade com o template oficial (4 variantes).
- **só re-splitar** (sem GPU/torch): `python ptbr_lora/tools/resplit.py` (dry-run) e `--write`
  (faz backup do split atual em `training/splits_backup_<data>/`). Escreve também `splits_meta.json`.

Locutores terminados em `:99` são buckets de locutores dissolvidos/não atribuídos: não servem de
referência e treinam só sem referência.

## Formato do `texts.csv`

Uma linha por item, no formato `wavs/<idx>.wav==texto`:

```
wavs/sample-1.wav==Ele trabalha no escritório desde o ano passado.
wavs/podcast_000123.wav==A previsão do tempo indica pancadas de chuva à tarde.
```

O parser aceita esse formato direto; `speakers.jsonl` (opcional) acrescenta
`{"idx","speaker"}`; `ref_map.jsonl` (`{"idx","ref_idx","cos"}`) é **legado** — o treino v3 não o usa (a
referência é sorteada por época entre os clipes do mesmo locutor).

## Conferir a pureza de locutor

Antes de treinar, confira se os clipes de cada locutor são mesmo a mesma pessoa (diarização errada, convidado no microfone do apresentador, vinhetas): `python ptbr_lora/tools/speaker_purity.py --folders <pasta> --out <saida>` (ou `--manifest` / `--training-dir`). Gera um `report.html` com exemplos para ouvir. Detalhes em [SPEAKER_PURITY.md](SPEAKER_PURITY.md).

## Próximo passo

[`TRAINING.md`](TRAINING.md).
