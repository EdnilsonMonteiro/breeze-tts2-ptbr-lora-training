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

> **⚠️ Caveat: o WER pune formatação do ASR, não pronúncia.** Quando o texto de
> referência traz números/siglas **por extenso** e a hipótese do ASR usa
> **algarismos**, o WER dispara sem erro de fala. Exemplos reais: ref
> "…*quinze oh dois*…" → hyp "…*1512*…" (WER 0,23); ref "…*CPF um dois três…*" →
> hyp "*CPFEI 123.456.789-01*" (WER 0,94). A **prosa** marca 0,000. Interprete o WER
> por faixa de frase (prosa vs números/siglas) — ver
> [`ESTRATEGIA-PTBR.md`](ESTRATEGIA-PTBR.md) §3.1.

## Similaridade de locutor (ECAPA)

Compara embeddings ECAPA (cosseno) entre referência(s) e áudio(s) gerado(s). Por
padrão, descobre automaticamente em `training/clone_out/` as referências
`ref_*.wav` e as gerações `*.wav` (exceto as referências):

```bash
python ptbr_lora/eval/spk_similarity.py
python ptbr_lora/eval/spk_similarity.py --dir <pasta> --refs a.wav b.wav --gens c.wav
```

Opções: `--dir`, `--refs`, `--gens`, `--device` (`cuda`/`cpu`), `--savedir`,
`--rms-match`, `--others`, `--wavlm`, `--exclude`.

- **`--rms-match`** (recomendado): iguala o RMS de todas as amostras antes de
  medir — remove o artefato "geração mais alta ⇒ cos menor" (no dado do projeto
  `r(cos, rms) = −0,40`).
- **`--others a.wav b.wav`** (controle obrigatório): imprime o **piso de outro
  locutor**. Se esse piso ficar perto do valor da geração, a métrica está saturada.
- **`--wavlm`**: segunda opinião (nas amostras do projeto mostrou-se **pouco
  discriminativa** — dá ~0,93 até para outro locutor — use com `--others`).

Regra prática de cosseno: **~0,7+** ≈ mesmo locutor; **<0,3** ≈ outra pessoa.
Sempre compare com o **teto intra-locutor** (cos entre duas gravações reais da
mesma pessoa: **0,825** no material do projeto) e reporte **mediana + desvio**
(nunca o máximo de N — é viés de seleção). Protocolo completo em
[`ESTRATEGIA-PTBR.md`](ESTRATEGIA-PTBR.md) §3.2 e §4.

## ⚠️ Como escolher o melhor modelo (protocolo corrigido)

As métricas **isoladas enganam** — no material deste projeto o **SECS premiou o
modelo base** (sem LoRA, que não fala pt-BR) e o **WER das amostras sem referência**
não mede o que se ouve. Use, na ordem:

1. **Controle do base.** Gere os mesmos clipes com o adapter desligado e meça com
   as mesmas ferramentas. Se o seu adapter não bater o base em WER, ele não aprendeu
   nada de útil.
2. **WER/CER nos clipes de CLONAGEM** (`reference/*.wav`), com `--normalize` e
   `--exclude` dos confusos de ASR (ex.: `regressao-en siglas-pt`). É a métrica que
   acompanha o ouvido.
3. **SECS apenas relativo ao base** e com `--others`. Lembre: o ECAPA mede
   **canal/timbre**, não pronúncia; um adapter mais treinado pode pontuar *menos*
   aqui sem soar pior.
4. **Medoid de N** (`gerar_amostras_referencia.py --candidates N`): remove a
   variância da amostra única antes de medir qualquer coisa.
5. **Audição cega** em última instância — é ela que decide.

Resumo do caso real (`r70_01` vs `r71_01` vs `r72_01`): o **`r72_01`** venceu no
**WER corrigido (0,117)** e no ouvido, apesar de ter o **menor SECS** — a explicação
completa está em [`ESTRATEGIA-PTBR.md`](ESTRATEGIA-PTBR.md) §3.2 e §3.4.

## Acompanhar o treino

```bash
tensorboard --logdir <PTBR_ARTIFACTS>/training/runs/<run>/tb
```

## Ouvir o resultado

Para julgamento perceptual (MOS informal) e clonagem, use o repo de inferência
[`breeze-tts2-ptbr`](https://github.com/EdnilsonMonteiro/breeze-tts2-ptbr):
`ui/app.py` (interface) ou `infer/clone_voice.py` (CLI). Nenhuma métrica
substitui a audição para timbre.
