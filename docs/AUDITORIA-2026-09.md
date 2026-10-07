# Auditoria 29/09/2026 — o que estava errado, o que mudou e o que ainda precisa de GPU

> **Atualização (10/2026):** os caminhos listados na seção 2 foram exercitados em GPU nas runs r74–r76
> (r76, passo 1500, é o adaptador liberado). A seção 2 e a 4 ficam como registro do que foi
> verificado na época; o protocolo v3 descrito aqui é o que o repositório implementa.

Revisão completa do treino (`ptbr_lora/`), das runs `r64_*`…`r72_01` e das métricas. Este
documento lista **achados → correção → como verificar**. O protocolo resultante é a **v3**
(este repo) e é **incompatível numericamente** com os números da v2 (val_loss, WER e SECS
antigos usavam split com vazamento, self-ref e métricas viesadas).

## 1. Achados e correções

| # | Achado | Efeito | Correção (v3) |
|---|---|---|---|
| 1 | **Split com vazamento de grupo.** Em `tagarela`/`podcast` o "locutor" é `programa:N` (diarização por episódio); o split era por esse id, então o **mesmo programa/voz caía em treino e em val/test** (179 grupos vazando, medido em `dataset_meta.jsonl`). | `val_loss` otimista e sem poder de detectar sobreajuste; "locutor não visto" não era verdade. | `core/splits.py`: unidade de split = **grupo** (`tagarela/podcast`: `speaker` sem o sufixo `:N`; demais: locutor), por corpus, por déficit, seed 42, com **auditoria** de vazamento (falha alto). `tools/resplit.py`. |
| 2 | **Self-ref (~30 %)** ("CPM") + referência **canônica fixa** por locutor. O alvo vazava para o prompt (o modelo aprende a copiar o áudio) e o modelo memorizava 1 clipe por locutor; a "canônica" documentada como *medoid* era na prática o `ref_idx` mais usado / menor sha1. | Aprende "repita o prompt" e "esta voz", não "clone uma voz nova". | `core/refs.py`: referência = **outro clipe do mesmo locutor, re-sorteado a cada época**; nunca self-ref; nunca canônica. Pool **do mesmo split** do item. |
| 3 | **Buckets `:99`** (locutores dissolvidos, ~12 mil itens ≈ 10 %) tratados como locutor: viravam pool de referência misturando vozes diferentes. | Sinal de identidade contraditório. | `eligible_ref_speaker`: `:99` nunca vira referência; itens `:99` treinam só sem referência (peso extra `--unassigned-weight 0.5`). |
| 4 | **Treino ≠ uso.** 100 % `ref_edit_tata`; a UI tem aba sem referência, o CFG negativo/dual usa `ref_clone_tata` e o texto de treino não era normalizado (a inferência normaliza). Instrução: pool de 9 frases, mas a inferência usa 1. | Modos de inferência fora da distribuição de treino. | Mix de condições (`--mix`, default 75 % `ref_edit` / 10 % `ref_clone` / 10 % `tts_instruction` / 5 % `tts_plain`), instrução **fixa** (`--instruction-mode fixed`), `text_norm` no treino (`--no-text-norm` desliga). |
| 5 | **Val**: pequeno (96 itens), com self-ref, média de médias por batch, sem separar backbone/depth, sem IC. | Ruído maior que as diferenças entre checkpoints. | Val **cross-ref**, 1 item/forward ponderado por frames supervisionados, backbone/depth separados, por corpus, 320 itens, `best/` salvo pelo val. `eval_val_full.py` com IC95. |
| 6 | **Amostras de treino sem referência** (`tts_instruction`) — tarefa diferente da usada em produção. | WER de checkpoint não representava o uso. | Amostras em `ref_edit_tata` com referência de locutor do **val**. |
| 7 | **WER** com `vad_filter=True`, beam 1, sem normalização por padrão, 3 cópias divergentes de `wer_cer`, frase em inglês misturada na média pt. | WER escondia arrasto/alucinação e punia formatação do ASR. | `core/asr_metrics.py` (única): normaliza ref e hyp (antes de minusculizar), S/D/I, falhas catastróficas, IC bootstrap; `eval_wer.py` sem VAD, beam 5, pt/en separados. |
| 8 | **SECS** contra o próprio prompt (viés de canal) e sem controle. | Inflava a identidade; premiou o base. | `spk_similarity --prompt-ref` (SECS *held-out*), IC bootstrap; `eval_zero_shot.py`: teto (real vs real), piso (outro locutor) e SECS normalizado. |
| 9 | **`text_norm`**: toda sigla em caixa alta era soletrada (`ATENÇÃO` → "a tê e ene…"), 1 grupo de milhar, sem moeda/%/hora com minutos/data/ordinal/romano/CPF. | Frases em caixa alta e números fora do dicionário eram lidos errado. | Reescrita em `core/text_norm.py` (fonte única, copiada p/ a UI) + testes. |
| 10 | **`apply_adapter_scale`**: cache do scaling-base só das chaves da 1ª chamada. | Adapter carregado depois ficava com escala errada/acumulada. | `core/adapter_scale.py` (base por chave, idempotente) + testes. |
| 11 | **Logging/scheduler**: `s_per_step` dividido por `grad_acc` numa janela de 5 passos; degrau de LR no fim de cada ciclo do cosseno com piso; passo parcial no fim da época; grad-clip 1.0 sem métrica; sem retomada do otimizador; `del pm` duplicado em `eval_val_full`. | Métricas de velocidade erradas, LR incorreto, runs não retomáveis. | `lr_schedule.py` (testado), passos por época = floor, `--max-grad-norm 5.0` + `clip_frac` no log, `trainer_state.pt` + `--resume-state`, skip de loss/grad não finitos. |
| 12 | Referência de inferência (UI) reamostrada a **48 kHz** sem trim; treino usa 24 kHz/trim/peak-norm. | Prompt fora do formato treinado. | `core/reference_prep.py` (modo `train` por padrão). |
| 13 | CRLF/LF misto (`git diff` acusando 700+ linhas). | Diffs ruidosos. | `.gitattributes` + working tree normalizada (backup em `_backup_pre_revisao_2026-09-29/`). |

## 2. O que NÃO tinha sido validado na época (exigia GPU)

Nada disto pôde ser executado aqui (sem GPU/torch). Passa por `py_compile` + pyflakes; os
módulos puros têm testes (`tests/ptbr`, `pytest`), mas **estes caminhos precisam de um smoke**:

1. `python ptbr_lora/core/train_lora.py --run smoke --smoke --steps 30` — exercita `TrainDataset`
   novo, condições, `build_item` e o loop. Esperado: imprime o `mix`, histograma de variantes,
   `val cross-ref: N itens` e roda 30 passos sem exceção.
2. Uma run curta com `--val-every-steps 20 --sample-every-steps 20 --state-every-steps 20 --steps` pequeno
   (`--smoke` não gera amostras): confere `generate_samples` com referência (`ref_edit_tata`
   via `prepare_inputs` + `generate(**inputs)`), o `best/` e o retomar com `--resume-state`.
3. `python ptbr_lora/eval/eval_zero_shot.py --adapter <a> --n 8 --out <dir>` antes de rodar `--n 200`.
4. `prepare_dataset.py finalize` (paridade oficial) — agora paridade em 4 variantes (com self-ref só nesse teste).

Se `generate(**inputs)` reclamar de chave inesperada nas amostras com referência (o caminho
sem referência continua como antes: remove `input_values`/`cfg_scale`), copie o tratamento de
`gen_core.generate_one`.

## 3. Como reverter

- Splits: `resplit.py --write` copia os arquivos atuais para `training/splits_backup_<data>/`;
  para voltar, copie de volta. Backup extra desta auditoria: `_backup_pre_revisao_2026-09-29/splits_v2/`.
- Código: nada foi commitado; `git diff` mostra tudo; `_backup_pre_revisao_2026-09-29/*.tgz` tem a árvore anterior.
- Runs antigas continuam carregando (o adapter não mudou), mas seus `val_loss` não são comparáveis
  com os da v3 — reavalie pelo `eval_val_full.py` sob o mesmo split.

## 4. Recomendação de sequência

1. `resplit.py` (dry-run → `--write`) → smoke de treino → `eval_zero_shot.py --adapter ''` (base) e `--adapter r72_01/final` (baseline v2 sob o protocolo novo).
2. Só então uma run v3 (`docs/TRAINING.md`), comparando pelo `eval_zero_shot` (macro WER, taxa de falha, SECS held-out normalizado) — não pelo val_loss isolado.
