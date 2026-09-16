# Avaliação

Três eixos: **generalização** (val loss), **inteligibilidade** (WER/CER) e
**identidade de voz** (similaridade de locutor).

## Val loss no conjunto completo

Avalia e ranqueia adapters com a mesma loss do treino (menor = melhor):

```bash
# base + adapters; <adapter> pode ser um nome em training/ ou um caminho
python ptbr_lora/eval/eval_val_full.py --adapters r64_02/checkpoints/step4000 --out results.json

# smoke (rápido)
python ptbr_lora/eval/eval_val_full.py --limit 64 --adapters <adapter> --out results.json
```

Opções: `--adapters` (lista), `--batch` (default 4), `--limit`, `--out`.
O base (sem adapter) é sempre avaliado primeiro como referência. A saída é um
JSON ordenado por `val_full`.

## WER/CER (inteligibilidade)

Transcreve uma pasta de WAVs gerados e compara com o texto de referência:

```bash
python ptbr_lora/eval/eval_wer.py --dir <pasta-de-wavs> --size large-v3 --device cpu
```

Opções: `--dir` (obrigatório), `--size` (default `large-v3`), `--device`
(`cpu`/`cuda`), `--compute-type` (default `int8`). Usa `faster-whisper` e lê os
textos de `training/dataset_meta.jsonl`.

## Similaridade de locutor (ECAPA)

Compara embeddings ECAPA (cosseno) entre referência(s) e áudio(s) gerado(s). Por
padrão, descobre automaticamente em `training/clone_out/` as referências
`ref_*.wav` e as gerações `*.wav` (exceto as referências):

```bash
python ptbr_lora/eval/spk_similarity.py
python ptbr_lora/eval/spk_similarity.py --dir <pasta> --refs a.wav b.wav --gens c.wav
```

Opções: `--dir`, `--refs`, `--gens`, `--device` (`cuda`/`cpu`), `--savedir`.
Regra prática de cosseno: **~0,7+** ≈ mesmo locutor; **<0,3** ≈ outra pessoa.

## Acompanhar o treino

```bash
tensorboard --logdir <PTBR_ARTIFACTS>/training/runs/<run>/tb
```

## Ouvir o resultado

Para julgamento perceptual (MOS informal) e clonagem, use o repo de inferência
[`breeze-tts2-ptbr`](https://github.com/EdnilsonMonteiro/breeze-tts2-ptbr):
`ui/app.py` (interface) ou `infer/clone_voice.py` (CLI). Nenhuma métrica
substitui a audição para timbre.
