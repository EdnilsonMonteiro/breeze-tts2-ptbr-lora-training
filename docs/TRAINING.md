# Treino

O trainer aplica **LoRA (PEFT)** sobre o Breeze TTS 2 congelado. O codec de áudio
é sempre congelado. O adapter é salvo em
`<PTBR_ARTIFACTS>/training/runs/<run>/checkpoints/<tag>/`.

## Smoke test

Valida loader, labels, VRAM e o fluxo de salvamento em poucos passos:

```bash
python ptbr_lora/core/train_lora.py --run smoke --smoke --steps 30
```

## Treino completo

Receita atual (**v3** — protocolo corrigido em [`AUDITORIA-2026-09.md`](AUDITORIA-2026-09.md);
os hiperparâmetros de LoRA seguem os da v2, ver [`ESTRATEGIA-PTBR.md`](ESTRATEGIA-PTBR.md)):

```bash
# 0) (uma vez) split por grupo, sem vazamento — dry-run mostra o antes/depois; --write faz backup
python ptbr_lora/tools/resplit.py
python ptbr_lora/tools/resplit.py --write

# 1) smoke e treino
python ptbr_lora/core/train_lora.py --run smoke --smoke --steps 30
python ptbr_lora/core/train_lora.py --run r80_01 --epochs 2 \
  --rank 64 --alpha 256 --targets all \
  --lr 3e-5 --lr-floor 0.15 --lr-cycles 3 \
  --batch 2 --grad-acc 16 --val-every-steps 1000 --sample-every-steps 500
```

Retomar exatamente de onde parou (adapter + otimizador + scheduler + posição na época):

```bash
python ptbr_lora/core/train_lora.py --run r80_01 --epochs 2 <mesmos args> \
  --resume-state <PTBR_ARTIFACTS>/training/runs/r80_01/checkpoints/resume
```

### Condições de treino (v3)

Cada exemplo é sorteado por `(época, idx)` numa **mistura de condições** (`--mix`):

| condição | template | default | referência |
|---|---|---|---|
| `ref_edit` | `ref_edit_tata` (ref + instrução + texto) | 75 % | **outro** clipe do mesmo locutor, re-sorteado a cada época |
| `ref_clone` | `ref_clone_tata` (ref + texto) | 10 % | idem (é o ramo negativo do CFG/dual) |
| `instruction` | `tts_instruction` | 10 % | — |
| `plain` | `tts_plain` | 5 % | — |

Nunca há self-ref; o pool de referências vem do **mesmo split** do item; buckets `:99`
(locutor não atribuído) nunca são referência. Itens sem referência possível caem em
`tts_instruction`. Instrução **fixa** ("Fale com clareza e naturalidade.", igual à produção)
e texto normalizado com `text_norm` (igual à inferência).

### Opções

| Flag | Default | Descrição |
|---|---|---|
| `--run` | (obrigatório) | nome do run (pasta em `training/runs/`) |
| `--smoke` / `--steps N` | — / 30 | modo curto de validação |
| `--epochs` | 3 | épocas |
| `--batch` / `--grad-acc` | 4 / 8 (`2/2` no smoke) | batch efetivo = batch × grad-acc |
| `--lr` | 2e-4 | learning rate (AdamW, warmup + cosseno) |
| `--lr-floor` / `--lr-cycles` | 0 / 3 | piso do LR (fração do pico) e nº de ciclos cosseno com restart |
| `--warmup` / `--weight-decay` | 50 / 0.01 | |
| `--max-grad-norm` | 5.0 | clip global; `clip_frac` no `log.csv` mostra quantos passos foram clipados |
| `--rank` / `--alpha` | 16 / 32 | LoRA. **Escala** = `alpha/r` (ou `alpha/√r` com `--use-rslora`); v2/v3: 64/256 = 4,0 |
| `--targets` | `attn` | `attn` = q/k/v/o · `all` = + gate/up/down |
| `--exclude-text-encoder` | off | não adapta o T5Gemma2 (só backbone + depth decoder) |
| `--attn` | `eager` | `eager` (validada) ou `sdpa` (mais rápida, não validada) |
| `--mix` | 0.75/0.10/0.10/0.05 | ex.: `ref_edit=0.75,ref_clone=0.1,instruction=0.1,plain=0.05` |
| `--ref-edit-frac` | — | LEGADO: F `ref_edit` + (1−F) `instruction` (ignorado se `--mix`) |
| `--instruction-mode` | `fixed` | `pool` = 9 frases variadas (não recomendado) |
| `--no-text-norm` | off | não normalizar o texto de treino |
| `--val-items` / `--val-every-steps` | 320 / 1000 | val cross-ref ponderado; `0` = só no fim da época |
| `--corpus-weights` | auto | JSON `{corpus: peso}` do sampler; `--unassigned-weight` (0.5) rebaixa `:99` |
| `--sample-every-steps` | 500 | checkpoint + amostras (com referência do val) |
| `--state-every-steps` | 500 | grava `checkpoints/resume/` (adapter + otimizador) |
| `--sample-ref-audio/--sample-ref-text` | 2 clipes do val | referência das amostras |
| `--resume-adapter` | — | continua de um adapter com otimizador NOVO |
| `--resume-state` | — | retoma tudo (`checkpoints/resume`) |
| `--no-tensorboard` / `--no-wer` | off | desativa TensorBoard / WER por checkpoint |

### Saídas do run

```
training/runs/<run>/
├─ config.json              # hiperparâmetros efetivos + params treináveis
├─ config.json              # + mix, git, splits_meta, protocolo
├─ log.csv                  # step, losses, val (total/backbone/depth), lr, grad_norm, clip_frac, VRAM, s/passo
├─ trainable_modules.txt
├─ run_summary.json         # melhor val, fração de clip, skips, tempo
├─ checkpoints/<tag>/       # adapter LoRA; `best/` (menor val), `final/`, `resume/` (+ trainer_state.pt)
├─ samples/                 # WAVs (ref_edit com ref do val) + samples_meta.json + wer.json
└─ tb/                      # TensorBoard  (tensorboard --logdir training/runs/<run>/tb)
```

## Runs encadeadas (`auto_train.py`)

Lança `train_lora.py` em rodadas com LR decaindo, parando em platô/crash:

```bash
python ptbr_lora/train/auto_train.py \
  --resume "<PTBR_ARTIFACTS>/training/runs/<run_anterior>/checkpoints/final" \
  --max-rounds 6 --epochs 2 --lr 3e-5 --decay 0.5 --epsilon 0.002 \
  --rank 64 --alpha 256 --targets all
```

> A receita **v2** usa `--lr-floor 0.15 --lr-cycles 3` no `train_lora.py`; o
> `auto_train.py` encadeia runs com LR decrescente e pode ser usado para refino
> após a run principal.

Para interromper entre rodadas, crie `<PTBR_ARTIFACTS>/training/AUTO_STOP`.

## Dicas

- Runs longos em 16 GB: use `--batch 2 --grad-acc 16` (mesmo batch efetivo 32,
  **metade do pico** — a v1 morreu com OOM em `batch 4 --grad-acc 8`) e *gradient
  checkpointing* (já ativo no trainer); monitore `vram_peak_gb` no `log.csv`.
- A perda tem dois ramos (`backbone_loss` + `depth_loss`, λ=1); ambos devem descer (o val
  reporta os dois separados).
- Compare runs pelo `eval_zero_shot.py` (ver `EVALUATION.md`), não só pelo `val_loss`: com o
  split novo (por grupo) o val_loss de runs antigas **não é comparável**.
- Passos por época = `floor(itens / (batch × grad-acc))`; nada de gradiente parcial no fim da época.
- O texto-alvo é **contexto** (label `-100`); só os tokens de áudio treinam.
