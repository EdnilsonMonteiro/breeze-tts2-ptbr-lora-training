---
language:
- pt
license: other
license_name: breezeblue-research-and-non-commercial-license
license_link: https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE
base_model: BreezeBlue/Breeze-TTS-2
library_name: peft
pipeline_tag: text-to-speech
tags:
- lora
- peft
- text-to-speech
- voice-cloning
- portuguese
- brazilian-portuguese
- pt-br
- breeze-tts
---

# Breeze-TTS-2 — Brazilian Portuguese LoRA (r76, step 1500)

A LoRA adapter that teaches [BreezeBlue/Breeze-TTS-2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) (an open-weight voice-cloning codec language model released for English and Mandarin) to speak **Brazilian Portuguese (pt-BR)**, cloning a voice from a short reference clip.

> **Derived from Breeze TTS 2 by BreezeBlue and licensed for research and non-commercial use only.**
> This adapter is a Derivative Model of Breeze-TTS-2 and is governed by the [BreezeBlue Research and Non-Commercial License](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE). Commercial use requires a separate licence from BreezeBlue. No affiliation with or endorsement by BreezeBlue is implied. See `NOTICE`.

- **Author:** Ednilson Monteiro
- **Inference code and UI:** [EdnilsonMonteiro/breeze-tts2-ptbr](https://github.com/EdnilsonMonteiro/breeze-tts2-ptbr)
- **Training code:** [EdnilsonMonteiro/breeze-tts2-ptbr-lora-training](https://github.com/EdnilsonMonteiro/breeze-tts2-ptbr-lora-training)
- **Preprint:** in preparation (case study of the training/evaluation pipeline); the link will be added here.

## What is in this repository

| File | Content |
|---|---|
| `adapter_model.safetensors` | LoRA weights (924 tensors, fp32, ~593 MB) |
| `adapter_config.json` | PEFT configuration |
| `NOTICE` | attribution and licence notice |

Only **one** checkpoint is released: step 1500 of the run called *r76* in the paper.

## Adapter details

| | |
|---|---|
| Base model | `BreezeBlue/Breeze-TTS-2` (3.48 B parameters) |
| Method | LoRA (PEFT 0.20), rank 64, alpha 128 (scale alpha/r = 2), dropout 0.05, no rslora |
| Targets | attention (q, k, v, o) **and** MLP (gate, up, down) in the three transformer stacks (text encoder, backbone, depth decoder); codec, embeddings, projector and heads untouched |
| Trainable parameters | 148.3 M (4.3 % of the base) |
| Training data | ~122 h, 66,982 clips, 1,608 voices (below) |
| Training | 1,500 optimiser steps (effective batch 32) of a 2,800-step schedule, warm-started from an attention-only adapter (r74, step 2000); single RTX 4060 Ti 16 GB; the learning rate had **not** been annealed at this checkpoint |
| Selection | chosen post hoc among checkpoints of the run on a small benchmark (see Evaluation) |

### Training data (v6 base, training split)

| Source | Hours | Voices | Licence of the source |
|---|---:|---:|---|
| TAGARELA (podcast audio, speaker ids inferred by diarisation) | 56.1 | 1,435 | CC BY-NC-SA 4.0 |
| CML-TTS (Portuguese, audiobooks) | 28.3 | 27 | CC BY 4.0 |
| CETUC (read speech) | 24.1 | 30 | see distributor |
| TTS-Portuguese corpus (single speaker) | 7.1 | 1 | see distributor |
| Podcast subset assembled by the author from publicly available episodes | 5.1 | 11 | as published by the creators; not redistributed |
| Common Voice pt | 1.7 | 104 | CC0 |

Training audio is **not** redistributed here. Sampling was by voice (sub-linear voice mass, per-voice and per-show caps), not by clip.

## How to use

The adapter needs the Breeze-TTS-2 engine; the easiest way is the inference repository, which downloads the base model and this adapter automatically:

```bash
git clone --recursive https://github.com/EdnilsonMonteiro/breeze-tts2-ptbr.git
cd breeze-tts2-ptbr && pip install -r requirements.txt

# command line
python infer/clone_voice.py \
  --adapter EdnilsonMonts/Breeze-tts-2-brazillian-lora \
  --ref-audio ref.wav --ref-text "exact transcript of the reference" \
  --text "Texto que o modelo deve falar." --out out/clone.wav

# web UI
python ui/app.py
```

Recommended practice (it is how the model was trained and evaluated):

- **Reference clip:** clean, 3–10 s, single speaker, with its **exact transcript**. Longer clips were never seen in training.
- **Text:** numbers, currency, times and acronyms must be spelled out as in training; the inference repository's normaliser (`core/text_norm.py`) does this. Split long text into blocks of at most ~10 s (the UI does).
- **Adapter scale:** keep it at 1.0 ("as trained").
- **Best-of-N (optional):** generate N candidates, discard implausible durations, keep those within 0.02 WER of the best (ASR), return the one most similar to the reference. The UI implements this rule.

## Evaluation (small, automatic only — read the limits)

Benchmark: fixed phrases, 4 seeds per phrase, WER with faster-whisper large-v3, speaker similarity (SECS) as ECAPA cosine against the reference recording (this score is biased upward by a shared channel; compare adapters only inside the same protocol). One training seed.

| Voices | N | SECS | WER | Generations with no word error |
|---|---|---:|---:|---:|
| 5 held-out voices (never in training; 2 phrases) | 1 | 0.705 | 0.001 | 98 % |
| same | 4 (best-of-4) | 0.747 | 0.000 | – |
| 3 reference voices (3 phrases) | 1 | 0.631 | 0.005 | 92 % |
| same | 4 (best-of-4) | 0.688 | 0.000 | – |

Compared with the attention-only adapter it was started from (r74 step 2000), this checkpoint makes **fewer word errors** (held-out: 98 % vs 55 % of generations error-free; the gap is concentrated on one probe phrase) and is **not detectably different in speaker similarity** on the held-out voices (paired difference −0.005, 95 % interval over voices [−0.044, +0.034], n = 5). It is not shown to be better than that adapter at best-of-4.

## Limitations

- The benchmark has only 3 + 5 voices and 2–3 sentences, no listening test, one training seed, and the checkpoint was selected post hoc; differences of ~0.02 SECS are within noise. Intervals computed per voice are 2–5× wider than a bootstrap over voice×phrase groups.
- "Held-out" means disjoint by *show/speaker group*; the same person appearing in another show cannot be excluded.
- Trained and tested on read and podcast speech; expressive style, emotion and very long texts are not evaluated. Behaviour in English/Chinese was not tested and is probably degraded.
- Pronunciation of rare words, foreign names and unusual numerals can still fail; use best-of-N or check the output.
- Training data include material of mixed provenance (crowd-sourced, audiobooks, podcasts); see the licences above.

## Responsible use

This is a **voice-cloning** model. Clone only voices you own or have explicit consent to use, and label synthetic audio as such. Impersonation, fraud, harassment and any other unlawful or harmful use are prohibited (as in the base model's licence). The reference voices used for testing are not distributed, and no audio of identifiable people is included in this repository.

## Citation

If you use this adapter, please cite the base model's repository ([breezeblue-ai/breeze-tts](https://github.com/breezeblue-ai/breeze-tts)) and this adapter:

```bibtex
@misc{monteiro2026breezeptbr,
  author = {Monteiro, Ednilson},
  title  = {Breeze-TTS-2 Brazilian Portuguese LoRA (r76, step 1500)},
  year   = {2026},
  howpublished = {\url{https://huggingface.co/EdnilsonMonts/Breeze-tts-2-brazillian-lora}}
}
```

## Acknowledgements

Breeze-TTS-2 by BreezeBlue (Apache-2.0 code, research/non-commercial weights); audio tokenizer based on Qwen3-TTS by the Alibaba Qwen Team (Apache-2.0); corpora: TAGARELA, CML-TTS, CETUC, TTS-Portuguese, Common Voice. Drafting and code review assisted by Claude (Anthropic); the author is responsible for the content.
