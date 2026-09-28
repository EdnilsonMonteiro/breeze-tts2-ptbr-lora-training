# PT-BR LoRA for Breeze TTS 2 — training (fork)

> **Official project (upstream):** [**breezeblue-ai/breeze-tts**](https://github.com/breezeblue-ai/breeze-tts)
> · **Model weights:** [BreezeBlue/Breeze-TTS-2](https://huggingface.co/BreezeBlue/Breeze-TTS-2)
> · **Model license:** [BreezeBlue Research and Non-Commercial](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE)

This repository is a **fork of the official [`breezeblue-ai/breeze-tts`](https://github.com/breezeblue-ai/breeze-tts)**
that adds the code used to train and evaluate a **Brazilian Portuguese (pt-BR)
LoRA adapter** on top of the Breeze TTS 2 codec language model, on a single
16 GB consumer GPU. For the original engine README, usage and full license, see
the official repository above.

- **Engine** (upstream, Apache-2.0): kept as-is at the repository root.
- **Project code** (this fork): `ptbr_lora/`.
- **Artifacts** (base weights, datasets, training runs): kept **outside** the git
  tree; see *Artifacts* below.

> Derived from Breeze TTS 2 by BreezeBlue and licensed for research and
> non-commercial use only. See `NOTICE`; the model license is at
> https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE.

> **Status — o modelo adaptador ainda NÃO foi publicado.** Este repositório
> contém apenas o **código** de treino/avaliação. O LoRA **não** está disponível
> aqui nem no Hugging Face; ele será liberado após treinos adicionais (mais runs).
> A publicação atual é apenas o **registro do código**.

> **Upstream commit:** baseado em `008f769` do
> [`breezeblue-ai/breeze-tts`](https://github.com/breezeblue-ai/breeze-tts)
> (ver [`UPSTREAM.md`](UPSTREAM.md) para sincronizar).

## Documentation

Public, user-facing documentation lives in [`docs/`](docs/README.md):

| Doc | Content |
|---|---|
| [`docs/INSTALL.md`](docs/INSTALL.md) | environment, dependencies, base checkpoint |
| [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) | `PTBR_ARTIFACTS` and all env vars |
| [`docs/DATASETS.md`](docs/DATASETS.md) | corpora download, ingestion, `prepare_dataset.py` |
| [`docs/TRAINING.md`](docs/TRAINING.md) | `train_lora.py`, `auto_train.py`, LoRA options |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | val loss, WER/CER, speaker similarity |
| [`docs/ESTRATEGIA-PTBR.md`](docs/ESTRATEGIA-PTBR.md) | **new recipe (v2)**: results, WER/SECS, why it improves consistency |

## Layout

```
ptbr_lora/
├─ core/      paths.py, common_breeze.py, prepare_dataset.py, train_lora.py
├─ data/      corpus download/ingestion (HF), speaker clustering (ECAPA)
├─ scraping/  podcast pipeline (YouTube -> 24 kHz chunks + transcription)
├─ train/     auto_train.py (chained runs)
├─ eval/      val_full, WER/CER, speaker similarity
└─ tools/     model/corpus diagnostics
docs/         public documentation (this repo)
```

## Install

```bash
git clone https://github.com/EdnilsonMonteiro/breeze-tts2-ptbr-lora-training.git
cd breeze-tts2-ptbr-lora-training
python -m venv venv && ./venv/Scripts/activate      # Windows (Linux: source venv/bin/activate)
pip install -r requirements.txt -r requirements-ptbr.txt
```

See [`docs/INSTALL.md`](docs/INSTALL.md) for details.

## Artifacts (outside git)

All paths come from `ptbr_lora/core/paths.py`, driven by environment variables.
Copy `.env.example` to `.env` and point `PTBR_ARTIFACTS` at the folder that holds
`datasets/`, `training/` and `models/`:

```
PTBR_ARTIFACTS=C:\IA\Breeze-tts
```

The base checkpoint is **not** included. Download `BreezeBlue/Breeze-TTS-2` from
Hugging Face into `<PTBR_ARTIFACTS>/models/Breeze-TTS-2`.

## Quick start

```bash
# 1) dataset (codes .npz, splits, gold samples)
python ptbr_lora/core/prepare_dataset.py process
python ptbr_lora/core/prepare_dataset.py finalize

# 2) smoke test + training (receita v2; ver docs/ESTRATEGIA-PTBR.md)
python ptbr_lora/core/train_lora.py --run smoke --smoke --steps 30
python ptbr_lora/core/train_lora.py --run r71_01 --epochs 2 \
  --rank 64 --alpha 256 --targets attn --ref-edit-frac 1.0 \
  --lr 3e-5 --lr-floor 0.15 --lr-cycles 3 --batch 2 --grad-acc 16

# 3) evaluation
python ptbr_lora/eval/eval_val_full.py --adapters <ckpt> --out results.json
```

See [`docs/TRAINING.md`](docs/TRAINING.md) and [`docs/EVALUATION.md`](docs/EVALUATION.md).

## Inference

To run the adapter (or the base model) through the web UI / CLI, use the
companion repository: **[EdnilsonMonteiro/breeze-tts2-ptbr](https://github.com/EdnilsonMonteiro/breeze-tts2-ptbr)**.

## Research status

**No trained model/adapter is released yet** — more runs are planned before any
adapter is published (Hugging Face or here).

The current recipe (**v2**, 20/09/2026) changed four things at once versus the
first runs (`r64_*`): classic LoRA scale `α/r=4.0` instead of `rsLoRA` `α/√r=8.0`;
an **LR floor** (15 %) with 3 cosine cycles so the LR never collapses to ~0;
100 % reference-conditioned examples with a **canonical reference per speaker** and
a cleaned identity signal (`REF_MIN_COS 0.70`, no self-ref); and `batch 2 × grad-acc
16` to avoid the OOM that killed v1.

**Best adapter so far: `r72_01`** (`--targets all`). It is the best in Portuguese and
the best voice by ear, and it wins the **corrected** metric too: measuring WER on the
actual cloning clips (normalized, ASR-confound sentences excluded) gives **0.117**
(`r72_01`) vs 0.147 (`r70_01`) and 0.167 (`r71_01`), against **0.908** for the frozen
base. Two metric traps were found and fixed: the training WER was measured on
reference-free samples (wrong task), and the **speaker-similarity (SECS) metric
favours the base model** (it tracks channel/timbre, not pronunciation), so it must be
read only relative to the base. Details, recipe and controls:
[`docs/ESTRATEGIA-PTBR.md`](docs/ESTRATEGIA-PTBR.md) §3.2–§3.4.

> Earlier caveat (superseded): in the v1 comparisons the reference-free and
> reference-conditioned runs also differed in learning rate, so that effect was
> reported as a joint data-and-optimizer intervention. The v2 runs isolate the
> variables (only `--targets` differs between `r71_01` and `r72_01`).

## License

- Engine code (upstream): Apache-2.0 ([`LICENSE`](LICENSE)) — from
  [breezeblue-ai/breeze-tts](https://github.com/breezeblue-ai/breeze-tts).
- Breeze TTS 2 weights/derivatives: BreezeBlue Research and Non-Commercial
  (https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE). Commercial
  use requires a separate license.
