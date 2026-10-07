# Avaliação

Três eixos: **generalização** (val loss), **inteligibilidade** (WER/CER) e **identidade de
voz** (similaridade de locutor). Protocolo **v3** (ver [`AUDITORIA-2026-09.md`](AUDITORIA-2026-09.md)):
val/test **por grupo** (sem vazamento), condição **cross-ref** (prompt = outro clipe do mesmo
locutor) e métricas com IC95. Números de runs antigas (`r64_*`…`r72_01`) foram medidos com
outro protocolo e **não são comparáveis** com os novos.

## Zero-shot no TEST (a avaliação que decide)

```bash
# 1) modelo base (controle) e o adapter; --n 8 primeiro como smoke
python ptbr_lora/eval/eval_zero_shot.py --adapter "" --n 200 --out <dir>/base
python ptbr_lora/eval/eval_zero_shot.py --adapter <run>/checkpoints/final --n 200 --out <dir>/r80_01
```

Gera (`ref_edit_tata`, instrução de produção, texto normalizado) o texto de itens do **test**
(locutores nunca vistos, estratificado por corpus) usando **outro clipe do locutor** como prompt e mede:

- **WER/CER** (Whisper large-v3, normalizado nos dois lados, sem VAD), S/D/I e **taxa de falha
  catastrófica** (WER > 0,5, duração fora de 0,4–2,5× a esperada, repetição);
- **SECS held-out** (ECAPA) contra clipes do locutor que **não** foram o prompt; **teto** (áudio real
  vs. real) e **piso** (geração vs. outro locutor); **SECS normalizado** = (gen − piso)/(teto − piso);
- macro (média por corpus) e por corpus, com IC95 bootstrap. Saídas: `results.json`, `items.csv`.

Retomável (não regera wavs existentes); `--phase gen|eval` separa geração de métricas.

## Val loss no conjunto completo

Val/test **cross-ref**, 1 item por forward, ponderado por frames supervisionados, backbone e depth
separados, por corpus, IC95 (diferenças dentro do IC não são reais):

```bash
python ptbr_lora/eval/eval_val_full.py --adapters r80_01/checkpoints/best --out results.json
python ptbr_lora/eval/eval_val_full.py --split test --adapters <adapter> --out results_test.json
python ptbr_lora/eval/eval_val_full.py --limit 64 --adapters <adapter> --out results.json   # smoke
```

Opções: `--adapters`, `--split` (`val`/`test`), `--limit`, `--out`. O base é avaliado primeiro.

## WER/CER (inteligibilidade)

```bash
python ptbr_lora/eval/eval_wer.py --dir <pasta-de-wavs> --size large-v3 --device cpu
```

Opções: `--dir`, `--size`, `--device`, `--compute-type`, `--match`, `--no-normalize`.
Frases de `core/sample_texts.py` (nomes `*-en` → `language="en"`, reportadas à parte).

O cálculo mora em `core/asr_metrics.py` (única implementação): `text_norm` aplicado em **ref e
hyp antes de minusculizar** (por isso "1512" e "quinze doze" empatam), S/D/I, decodificação
**sem** `vad_filter`/`condition_on_previous_text` (o VAD escondia arrasto e alucinação) e
falhas catastróficas. Ignora o caveat antigo "WER pune formatação": ele só valia sem normalização.

## Similaridade de locutor (ECAPA)

```bash
python ptbr_lora/eval/spk_similarity.py --dir <pasta> --refs prompt.wav outra_gravacao.wav \
    --prompt-ref prompt.wav --rms-match --others outro_locutor.wav
```

Opções: `--dir`, `--refs`, `--gens`, `--prompt-ref`, `--device`, `--savedir`, `--rms-match`,
`--others`, `--wavlm`, `--exclude`.

- **`--prompt-ref`** (importante): marca as refs que foram usadas como **prompt** na geração. O SECS
  contra o prompt é **viesado** (o modelo copia canal/sala); o número que vale é o **held-out**
  (contra as demais gravações do locutor). Sem nenhuma ref held-out o script avisa.
- **`--rms-match`**: iguala o RMS antes de medir (`r(cos, rms) = −0,40` no dado do projeto).
- **`--others`**: piso de outro locutor. Se ficar perto da geração, a métrica está saturada.
- **`--wavlm`**: segunda opinião (pouco discriminativa neste material — use com `--others`).

Cosseno: **~0,7+** ≈ mesmo locutor; **<0,3** ≈ outra pessoa; compare com o **teto intra-locutor**
(0,825 no material do projeto) e reporte **média + IC95** (nunca o máximo de N).

## ⚠️ Como escolher o melhor modelo (protocolo corrigido)

As métricas **isoladas enganam** — no material deste projeto o **SECS premiou o
modelo base** (sem LoRA, que não fala pt-BR) e o **WER das amostras sem referência**
não mede o que se ouve. Use, na ordem:

1. **Controle do base.** Gere os mesmos clipes com o adapter desligado e meça com
   as mesmas ferramentas. Se o seu adapter não bater o base em WER, ele não aprendeu
   nada de útil.
2. **WER/CER nos clipes de CLONAGEM** e no `eval_zero_shot` (normalização já é o default;
   `*-en` sai da média pt automaticamente). É a métrica que acompanha o ouvido.
3. **SECS apenas relativo ao base** e com `--others`. Lembre: o ECAPA mede
   **canal/timbre**, não pronúncia; um adapter mais treinado pode pontuar *menos*
   aqui sem soar pior.
4. **Medoid de N** (`gerar_amostras_referencia.py --candidates N`, `gerar_em_blocos.py`): remove a
   variância da amostra única para PRODUZIR áudio. Para AVALIAR um modelo não use medoid (ele
   esconde a variância real): meça todas as amostras e reporte média, IC e taxa de falha.
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
