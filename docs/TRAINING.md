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

Receita atual (**v2** — ver [`ESTRATEGIA-PTBR.md`](ESTRATEGIA-PTBR.md)):

```bash
python ptbr_lora/core/train_lora.py --run r71_01 --epochs 2 \
  --rank 64 --alpha 256 --targets attn \
  --ref-edit-frac 1.0 --lr 3e-5 --lr-floor 0.15 --lr-cycles 3 \
  --batch 2 --grad-acc 16 --val-items 96 --sample-every-steps 500
```

Exemplo v1 (histórico):

```bash
python ptbr_lora/core/train_lora.py --run r64_02 --epochs 2 \
  --rank 64 --alpha 64 --targets all --use-rslora \
  --ref-edit-frac 0.9 --lr 3e-5 --val-items 96
```

### Opções

| Flag | Default | Descrição |
|---|---|---|
| `--run` | (obrigatório) | nome do run (pasta em `training/runs/`) |
| `--smoke` / `--steps N` | — / 30 | modo curto de validação |
| `--epochs` | 3 | épocas |
| `--batch` / `--grad-acc` | 4 / 8 (`2/2` no smoke) | batch efetivo = batch × grad-acc |
| `--lr` | 2e-4 | learning rate (AdamW, warmup + cosseno) |
| `--lr-floor` | off | fração **mínima** do LR que o cosseno não pode cruzar (ex.: `0.15` = 15 % do pico). Evita o LR colapsar a ~0 e "congelar" o depth decoder |
| `--lr-cycles` | 1 | nº de ciclos cosseno por run (ex.: `3`); recicla o LR em vez de decair uma única vez |
| `--warmup` | 50 | passos de warmup |
| `--rank` / `--alpha` | 16 / 32 | LoRA rank e alpha. **Escala efetiva** = `alpha/r` (ou `alpha/√r` com `--use-rslora`). A v2 usa `--rank 64 --alpha 256` (escala 4,0) **sem** `--use-rslora` |
| `--targets` | `attn` | `attn` = q/k/v/o · `all` = + gate/up/down (MLPs) |
| `--use-rslora` | off | escala rank-stabilized (`α/√r`). A v2 **não** usa (escala clássica `α/r`) |
| `--ref-edit-frac` | 0.9 | fração de exemplos com referência de locutor |
| `--val-items` | 96 | itens da validação estratificada durante o treino |
| `--corpus-weights` | auto | JSON `{corpus: peso}` (default: oversampling 24 kHz ×2) |
| `--no-tensorboard` / `--no-wer` | off | desativa TensorBoard / WER por checkpoint |
| `--sample-every-steps` | 500 | intervalo de amostragem de áudio |
| `--resume-adapter` | — | pasta de adapter para continuar o treino |

### Saídas do run

```
training/runs/<run>/
├─ config.json              # hiperparâmetros efetivos + params treináveis
├─ log.csv                  # step, losses, val_loss, lr, VRAM, s/step
├─ trainable_modules.txt
├─ checkpoints/<tag>/       # adapter LoRA (adapter_config.json + adapter_model.safetensors)
├─ samples/                 # WAVs de validação por checkpoint + wer.json
└─ tb/                      # TensorBoard  (tensorboard --logdir training/runs/<run>/tb)
```

### Títulos das variantes de treino

- `tts_instruction`: `[instrução] + [texto] + [áudio-alvo]` (sem referência).
- `ref_edit_auto` / `ref_edit_tata`: `[texto-ref, áudio-ref, texto] + [áudio-alvo]`
  (clonagem com referência do mesmo locutor quando disponível).
- A fração com referência é controlada por `--ref-edit-frac`.

## Runs encadeadas (`auto_train.py`)

Lança `train_lora.py` em rodadas com LR decaindo, parando em platô/crash:

```bash
python ptbr_lora/train/auto_train.py \
  --resume "C:\IA\Breeze-tts\training\runs\r71_01\checkpoints\final" \
  --max-rounds 6 --epochs 2 --lr 3e-5 --decay 0.5 --epsilon 0.002 \
  --rank 64 --alpha 256 --targets attn --ref-edit-frac 1.0
```

> A receita **v2** usa `--lr-floor 0.15 --lr-cycles 3` no `train_lora.py`; o
> `auto_train.py` encadeia runs com LR decrescente e pode ser usado para refino
> após a run principal.

Para interromper entre rodadas, crie `<PTBR_ARTIFACTS>/training/AUTO_STOP`.

## Dicas

- Runs longos em 16 GB: use `--batch 2 --grad-acc 16` (mesmo batch efetivo 32,
  **metade do pico** — a v1 morreu com OOM em `batch 4 --grad-acc 8`) e *gradient
  checkpointing* (já ativo no trainer); monitore `vram_peak_gb` no `log.csv`.
- A perda tem dois ramos (`backbone_loss` + `depth_loss`, λ=1); ambos devem descer.
- O texto-alvo é **contexto** (label `-100`); só os tokens de áudio treinam.
