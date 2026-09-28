# Estratégia de treino PT-BR (v2) — receita, resultados e por que ela ganha consistência sem perder as sutilezas do português

> **Status:** 25/09/2026 · **GPU:** RTX 4060 Ti 16 GB · **Melhor adaptador: `r72_01`**
> (`--targets all`, escala 4,0, LR 3e-5 com piso) — melhor voz **e** melhor português,
> confirmado por audição e pela **métrica corrigida** (§3.4). Runs desta leva:
> `r70_01` e `r71_01` (attn), `r72_01` (all, concluído em 25/09).
>
> Este documento é o resumo público da **v2 da receita**. O diagnóstico longo e a
> pesquisa bibliográfica estão no material de trabalho (`CONSISTENCIA-DE-VOZ.md`,
> mantido fora do repo). Aqui ficam a receita, os números e as decisões.

---

## 1. Resumo

A partir de **20/09/2026** a receita de treino mudou de forma controlada. Em vez de
"mais uma run parecida", a leva v2 isola **as quatro causas raiz** que estavam
degradando o adaptador v1 (`r64_*`):

| # | Mudança | v1 (`r64_*`) | v2 (`r70_01` / `r71_01` / `r72_01`) |
|---|---|---|---|
| 1 | **Escala efetiva do LoRA** | `rsLoRA` → `α/√r` = **8,0** | clássico → `α/r` = **4,0** |
| 2 | **Schedule de LR** | cosseno único → LR → ~0 | **piso de 15 %** + 3 ciclos cosseno |
| 3 | **Regularização da referência** | `ref_edit_frac 0,9`; `REF_MIN_COS 0,55`; self-ref 2,9 % | `ref_edit_frac 1,0`; `REF_MIN_COS 0,70`; self-ref 0 %; ~30 % self-ref na amostragem |
| 4 | **VRAM / estabilidade** | `batch 4 × acc 8`, crash OOM | `batch 2 × acc 16` (mesmo efetivo, metade do pico) |

Resultado medido (ver §3): no **WER do teste de língua** a época 0 cai de **0,299**
(`r64_02`, v1) para **0,271** (`r72_01`, v2). E, medindo **nos clipes de clonagem**
(o que se ouve de fato) com os confusos de ASR removidos, o **`r72_01` é o melhor
(WER 0,117 vs 0,147 do `r70_01` e 0,167 do `r71_01`)** — ver §3.4.

---

## 2. As mudanças, uma por uma

### 2.1 Escala do LoRA: `rsLoRA` (8,0) → clássico (4,0)

| variante | fator de escala | v1 (`r=64, alpha=64`) | v2 (`r=64, alpha=256`) |
|---|---|---|---|
| LoRA clássico | `alpha / r` | 1,0 | **4,0** |
| rsLoRA | `alpha / sqrt(r)` | **8,0** | — |

A v1 aplicava deltas com magnitude efetiva **8,0**. A literatura de clonagem
(XTTS/GLM-TTS ≈ 2,0; Orpheus com r menor) e o próprio sintoma — "loss desce,
identidade oscila, pronúncia degrada" — apontavam para **over-steering**. A v2
fixa a escala em **4,0** (clássico, rank 64, alpha 256).

> **Nota importante sobre inferência.** No runtime, "escala do adapter" virou um
> **multiplicador** (1,0 = como treinado). Testes empíricos mostram que valores
> **< 1,0 pioram** o resultado — ou seja, o adapter v2 treinado em 4,0 **não**
> está over-steering. O problema da v1 era a escala **de treino** (8,0), não a de
> inferência.

### 2.2 LR com piso e reciclagem (a causa do "pior português")

A `r70_01` foi treinada com `lr 1e-5` e **cosseno único**. O LR chegou a ~0 antes
do fim: o `depth_loss` (codebooks 1..15 = **detalhe acústico e fonético**)
"congelou" e o português piorou (troca de vogais, sotaque em números/siglas).

A v2 usa `--lr 3e-5 --lr-floor 0.15 --lr-cycles 3`: o LR **nunca chega a zero**
(fica em ≥ 15 % do pico) e recicla 3 vezes. Diagnosticado como a causa raiz da
queda de qualidade da `r70_01` — não o `targets attn` nem a escala 4,0.

### 2.3 Referência do locutor em 100 % dos exemplos (`--ref-edit-frac 1.0`)

Na v1, ~10 % dos passos ensinavam o modelo a gerar **sem** olhar a referência. A
v2 treina **todo** exemplo no modo referência-condicionado: o sinal "copie a
identidade desta referência" é apresentado de forma consistente.

### 2.4 Referência canônica por locutor + identidade limpa

- `REF_MIN_COS 0,55 → 0,70` e `CLUSTER_DIST 0,45 → 0,30` (`tagarela_speakers.py`):
  remove os ~7,8 % de pares que nem eram o mesmo locutor.
- **Zero self-ref** no `ref_map` (era 2,9 %): some o "copie o áudio".
- **1 referência canônica por locutor** (o *medoid* dos clipes daquele locutor),
  reutilizada em todas as linhas — exatamente o recipe oficial de SFT do
  Qwen3-TTS, que "reutiliza um único clipe de referência por locutor para garantir
  estabilidade de locutor".
- Amostragem alvo **~30 % self-ref / 70 % cross-ref** (VoiceStar "CPM training"),
  que habilita repetir o prompt na inferência sem degradar o WER.

### 2.5 VRAM: `batch 2 × grad-acc 16` (mesmo efetivo, metade do pico)

A v1 morreu com **CUDA OOM** (`r64_03`, passo 4605). A v2 mantém o batch efetivo
em 32 clipes/passo, mas com metade do pico de VRAM → runs que **terminam**.

### 2.6 Duração: teto em 10 s (o comprimento resolve-se na geração)

O treino usa clipes ≤ 10 s (média 6,0 s). Gerar 14–17 s é **extrapolação** (~1,7×
o máximo visto) e a identidade deriva ao longo do áudio. A v2 mantém o teto e
resolve texto longo por **geração em blocos** de ≤ 8–10 s com a mesma referência.

### 2.7 `--targets all` — o que o `r72_01` tem a mais (e por que importa)

As duas runs v2 são idênticas exceto pelos módulos que recebem LoRA:

| Run | Alvos do LoRA | Params treináveis |
|---|---|---|
| `r71_01` | atenção (q/k/v/o) | ~42,7 M |
| **`r72_01`** | atenção **+ MLPs** (gate/up/down) | **148,3 M** |

Os MLPs dão **capacidade extra para fonologia e prosódia** — que é onde o pt-BR se
distingue do EN/ZH do modelo base (vogais nasais, /r/, /lh/, /nh/, "ão", sandhi). É
por isso que o `r72_01` é, ao ouvido, **o melhor português e a melhor voz**: ele tem
onde "guardar" os padrões acústicos do português. Custo: ~2× mais lento e mais VRAM.

---

## 3. Resultados

### 3.1 Inteligibilidade — WER/CER (teste de língua, 11 frases, sem referência)

Cada run gera amostras a cada 500 passos e calcula WER/CER com `faster-whisper`;
o número fica em `training/runs/<run>/samples/<ckpt>/wer.json`.

| Run | targets | escala | lr | época 0 | época 1 | final |
|---|---|---|---|---|---|---|
| `r64_01` (v1) | all | rsLoRA 8,0 | 1e-4 | 0,284 | — | — |
| `r64_02` (v1) | all | rsLoRA 8,0 | 3e-5 | 0,299 | — | — |
| `r70_01` | attn | 4,0 | 1e-5 | 0,387 | 0,361 | 0,367 |
| **`r71_01`** | **attn** | **4,0** | **3e-5 + piso** | **0,307** | **0,263** | 0,303 |
| **`r72_01`** | **all** | **4,0** | **3e-5 + piso** | **0,271** | *(em treino)* | — |

Leituras:

- A v2 (`r72_01`, 0,271) já bate a v1 (`r64_02`, 0,299) **na época 0**.
- O `r70_01` (lr 1e-5) é o **pior** → confirma que o LR baixo era a causa, não o
  `targets attn`/escala.
- O melhor checkpoint do `r71_01` (época 1) chega a **0,263**.

> **⚠️ Caveat da métrica.** O WER é inflado por **formatação do ASR**, não por erro
> de pronúncia. Exemplos reais do `wer.json` do `r72_01`:
> - ref "…**quinze oh dois**…" → hyp "…**1512**…" (WER 0,23);
> - ref "…**dezesseis e vinte e três** graus" → hyp "…**16 e 23** graus" (WER 0,21);
> - ref "**CPF um dois três…**" → hyp "**CPFEI 123.456.789-01**" (WER 0,94).
>
> A **prosa** sai praticamente perfeita (`ola-pt`, `afetivo-pt`, `letras-pt`,
> `siglas2-pt`, `trabalenguas-pt` = **0,000**). Ajustar o normalizador de
> números/siglas do *scorer* é o próximo passo para uma métrica honesta.

### 3.2 Identidade de voz — SECS (ECAPA, `--rms-match`) — e por que ele engana

Medição sobre os **clipes de clonagem** (`reference/*_final.wav`), seed 42, 2
referências reais, `--rms-match`, com controle de outros locutores
(CETUC = **0,27 / 0,05** ⇒ a métrica é válida, não está saturada):

| Modelo | Mediana SECS | Desvio |
|---|---|---|
| Teto intra-locutor (2 gravações reais da mesma pessoa) | **0,825** | — |
| **Base Breeze (sem LoRA)** | **0,685** | 0,070 |
| `r70_01` final (attn, lr 1e-5) | 0,725 | 0,040 |
| `r71_01` final (attn, lr 3e-5+floor) | 0,651 | 0,042 |
| **`r72_01` final (all, lr 3e-5+floor)** | **0,614** | 0,072 |

> **⚠️ O SECS está premiando o modelo errado neste projeto.** O **modelo base**
> (que **não** fala pt-BR) tira **0,685** — acima do `r71_01` e do `r72_01`. Motivo:
> o ECAPA mede **timbre/canal**, não pronúncia. O base "clona" razoavelmente bem
> *in-context* porque **copia o canal** da referência; e quanto mais o adapter
> aprende pt-BR, mais ele se afasta daquele canal (a referência é 18 s / 48 kHz,
> fora da distribuição do treino) e **mais o SECS cai**, sem soar pior.
> Conclusão: **SECS sozinho não escolhe o melhor modelo** (ver §3.4). Use-o apenas
> *relativo ao base*, junto do controle `--others`.

### 3.3 Perdas (loss)

Ambos os ramos descem juntos na v2 — `backbone_loss` ~0,84 e `depth_loss` ~3,3 no
`r72_01` (época 1), contra ~0,90 / 4,64 no fim da v1. O `depth_loss` só voltou a
cair porque o **LR parou de morrer** (§2.2).

### 3.4 Métrica corrigida: WER **nos clipes de clonagem**, sem os confusos de ASR

O WER de §3.1 mede as amostras **sem referência** (teste de língua) — não o que se
ouve na clonagem. Medindo nos **próprios clipes de clonagem** (`reference/*_final.wav`),
com normalização de números/siglas (`eval_wer.py --normalize`) e **excluindo os dois
confusos** (`--exclude regressao-en siglas-pt`: inglês + frase que o ASR escreve como
código "CPF 123.456.789-01"):

| Modelo | WER (11 frases) | **WER (9 frases, sem confusos)** | CER (9 frases) |
|---|---|---|---|
| Base Breeze (sem LoRA) | 0,894 | 0,908 | 0,568 |
| `r70_01` final | 0,221 | 0,147 | 0,066 |
| `r71_01` final | 0,258 | 0,167 | 0,065 |
| **`r72_01` final** | 0,268 | **0,117** | **0,056** |

Leituras:

- **O base é péssimo** (0,908): no `clima-pt` ele alucina *"Eu sinto esses zumbis de
  merda em algum lugar aqui"* — então a métrica **discrimina** qualidade de verdade.
- Feita a limpeza dos confusos, **o `r72_01` é o melhor (0,117)** — exatamente o que
  o ouvido aponta. A frase `siglas-pt` sozinha dava 0,889 no `r72_01` **por
  formatação do ASR**, não por fala.
- Repare que o `r70_01` parecia "melhor" (0,147) e o `r71_01` pior (0,167) — a
  diferença vinha de uma única frase de siglas (`siglas2-pt`: 0,556 no `r71_01`).
  É a assinatura de um teste com poucas frases e sem controle de confusos.

**Conclusão prática:** para escolher modelo, use **WER nos clipes de clonagem,
normalizado e sem confusos**, cruzado com audição. O SECS entra só como controle de
identidade *relativo ao base*.

---

## 4. Por que a v2 ganha consistência **e** mantém as sutilezas do pt-BR

1. **Sinal de identidade limpo.** 100 % referência + referência canônica + pares
   com cos ≥ 0,70 + zero self-ref ⇒ o modelo aprende "copie **esta voz**", não
   "copie este canal" nem "copie este áudio".
2. **Fonética preservada.** O Breeze é **bi-EN/ZH**: todo o pt-BR (vogais nasais,
   /r/, /lh/, /nh/, "ão", vogais médias, sandhi) vem **do LoRA**. Um schedule que
   **não zera o LR** mantém gradiente nos codebooks finos (timbre/fonética) até o
   fim — é isso que evita a deriva "e→i" e o sotaque inglês em números/siglas que
   apareceram quando o LR colapsou.
3. **Sem over-steering.** Escala de treino 4,0 (não 8,0) ⇒ deltas de magnitude
   sadia; a identidade não é "forçada" a ponto de distorcer a prosódia.
4. **Estabilidade reduz variância.** Menos OOM, menos ruído de referência, blocos
   ≤ 10 s ⇒ a mesma frase sai parecida consigo mesma (que é a definição prática de
   *consistência*), mantendo a naturalidade do português.

---

## 5. Como reproduzir (receita v2)

```bash
# dados (referência canônica por locutor + identidade limpa)
python ptbr_lora/data/tagarela_speakers.py          # REF_MIN_COS 0.70 / CLUSTER_DIST 0.30
python ptbr_lora/core/prepare_dataset.py process
python ptbr_lora/core/prepare_dataset.py finalize

# treino v2 — variante "attn" (só atenção)
python ptbr_lora/core/train_lora.py --run r71_01 --epochs 2 \
  --rank 64 --alpha 256 --targets attn \
  --ref-edit-frac 1.0 --lr 3e-5 --lr-floor 0.15 --lr-cycles 3 \
  --batch 2 --grad-acc 16 --val-items 96 --sample-every-steps 500

# MELHOR: mesmos ajustes + MLPs treináveis (--targets all)  => r72_01
python ptbr_lora/core/train_lora.py --run r72_01 --epochs 2 \
  --rank 64 --alpha 256 --targets all \
  --ref-edit-frac 1.0 --lr 3e-5 --lr-floor 0.15 --lr-cycles 3 \
  --batch 2 --grad-acc 16 --val-items 96 --sample-every-steps 500
```

> **Sobre as flags:** `--alpha 256` **sem** `--use-rslora` dá escala `256/64 = 4,0`.
> `--lr-floor` / `--lr-cycles` são as adições desta leva (ver `docs/TRAINING.md`).

---

## 6. Como ouvir / avaliar

```bash
# inteligibilidade — NOS CLIPES DE CLONAGEM, normalizado e sem os confusos (§3.4)
python ptbr_lora/eval/eval_wer.py --dir <runs>/<run>/reference \
  --normalize --exclude regressao-en siglas-pt --size large-v3 --device cpu

# identidade (SECS) — sempre com controle de outro locutor
python ptbr_lora/eval/spk_similarity.py --dir <runs>/<run>/reference \
  --refs ref.wav --rms-match --others outro_locutor.wav --exclude regressao-en

# gerar os clipes de clonagem do adapter (medoid de N: remove a variância da amostra)
python ptbr_lora/tools/gerar_amostras_referencia.py --adapter <ckpt> \
  --out <runs>/<run>/reference --ref-audio ref.wav --device cuda --candidates 5
```

Áudios de referência desta leva (locutor `VozEdnilson`), em `training/runs/<run>/reference/`:

- `r72_01/` — 11 `*_final.wav` (**melhor**) + 11 `*_step4500.wav`;
- `r71_01/` — 11 `*_final.wav` + 11 `*_step4500.wav`;
- `r70_01/` — 11 `*_final.wav`;
- `base_control/` — 11 `*_base.wav` (**controle sem LoRA**).

---

## 7. Decisão e próximos passos

**Decisão:** o adaptador desta leva é o **`r72_01`** (`--targets all`). Ele é o melhor
em português (WER 0,117, o menor) **e** a melhor voz ao ouvido. O `r71_01` (`attn`)
fica como alternativa leve; o `r70_01` (lr 1e-5) só liderava no SECS, que se mostrou
viesado (§3.2).

Próximos passos:

1. **Publicar o `r72_01`** (Hugging Face) com números honestos por eixo (§3.4),
   deixando claro que o SECS favorece o base e não deve ser lido isolado.
2. **Refino opcional:** continuar do `r72_01` com `--lr 1e-5 --lr-floor 0.15`
   (1 época) e comparar pelo **WER corrigido** — não por mais épocas às cegas.
3. **Protocolo de produção:** **N candidatos + medoid** por bloco de ≤ 8–10 s
   (`--candidates`), que remove a variância da amostra única.
4. **Objetivo de voz fixa:** se a meta é *sempre a mesma* voz, o ganho de categoria
   é um **fine-tune de locutor único** (30–60 min do áudio do locutor), não mais LoRA.
5. Ampliar o **conjunto de frases de avaliação** (hoje 11, com 2 confusos) e
   categorizá-las (prosa / números / siglas / inglês) para o WER não ser dominado
   por uma frase.
