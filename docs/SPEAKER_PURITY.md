# Pureza de locutor — os clipes de cada voz são mesmo a mesma pessoa?

`ptbr_lora/tools/speaker_purity.py` audita um dataset de voz **antes do treino**. Para cada rótulo de
locutor, mede com embeddings de voz (ECAPA) se todos os clipes soam como a mesma pessoa. Também aponta
clipes intrusos, rótulos que misturam duas vozes e rótulos diferentes que são a mesma pessoa.

**Por que importa:** o treino de clonagem usa "outro clipe do mesmo locutor" como referência. Se o rótulo
junta duas pessoas (erro comum de diarização em podcast, entrevista ou vídeo), o modelo aprende que a
referência não é confiável e passa a ignorá-la.

## Uso rápido

Funciona com o venv do projeto (precisa de `speechbrain`, `soundfile` e `scipy`). Na primeira vez,
baixa o ECAPA (~80 MB) se ele ainda não estiver em `dataScrapping/work/spkrec-ecapa`.

**1. Seu próprio áudio, uma pasta por pessoa.** É o jeito mais simples.

```
meus_audios/
  joao/   01.wav 02.wav ...
  maria/  a.wav  b.flac ...
```
```bash
python ptbr_lora/tools/speaker_purity.py --folders meus_audios --out purity_meus
```

**2. Qualquer manifesto (CSV, TSV, `|` ou JSONL).** Os campos de cada linha entram em modelos entre `{}`:

```bash
# chunks de podcast diarizados: o rótulo é episódio + SPEAKER_xx
python ptbr_lora/tools/speaker_purity.py --manifest dataScrapping/work/chunks_index.jsonl \
  --root datasets/podcast/wavs --audio "{chunk}" --speaker "{tag}:{speaker}" --group "{tag}" \
  --out purity_podcast
```

**3. Pasta de treino do projeto (preset).** Lê `manifest.csv`, o split de `speaker_table.csv` e `wavs24/`:

```bash
python ptbr_lora/tools/speaker_purity.py --training-dir <PTBR_ARTIFACTS>/training_v6 --out purity_v6
```

Dica: `--only-group podcast` restringe a um grupo. Se um treino estiver rodando, use `--device cpu` para
não disputar a GPU.

## O que sai em `--out`

| Arquivo | Conteúdo |
|---|---|
| `report.html` | Resumo por corpus e **exemplos para ouvir** dos piores locutores (clipe típico × clipe que destoa, ou grupo A × grupo B) |
| `report.md` | O mesmo resumo em texto |
| `speakers.csv` | Uma linha por locutor: teto, outliers, teste de 2 vozes, locutor mais parecido, status e motivos |
| `clips.csv` | Uma linha por clipe: similaridade com o próprio locutor e se é outlier |
| `duplicates.csv` | Pares de rótulos com a mesma voz; `cross_split=1` = **vazamento** treino↔validação |
| `exclude_speakers.txt` | Locutores `RUIM`, um por linha |
| `exclude_clips.txt` | Clipes que destoam em locutores não-`RUIM` |
| `summary.json` | Números por grupo e configuração usada |
| `embeddings.npz` | Cache: rodar de novo só calcula o que falta |

## Como ler

- **Teto (intra):** média da similaridade entre pares de clipes do mesmo rótulo. Com ECAPA, estúdio limpo
  fica em ~0,65–0,80, gravações caseiras um pouco abaixo, e menos de ~0,45 raramente é uma pessoa só.
  **Compare o teto mediano entre corpora:** um corpus bem abaixo dos outros tem rótulos menos confiáveis.
- **Destoa (outlier):** clipe bem abaixo da mediana do próprio locutor (critério robusto, MAD). Costuma ser
  outra pessoa, vinheta, risada ou fala sobreposta.
- **Parece 2 vozes:** o rótulo se divide em dois grupos coesos e diferentes entre si. É o sinal típico de
  diarização que juntou duas pessoas.
- **Igual a X:** dois rótulos com a mesma voz, como o mesmo apresentador em episódios diferentes. Entre
  splits diferentes, o resultado de validação fica otimista (vazamento).

Status: `RUIM` (teto baixo, muitos intrusos ou 2 vozes), `SUSPEITO` (no limite ou duplicata), `OK`,
`POUCOS` (menos de `--min-clips` clipes; sem veredito).

**Ouça antes de excluir.** Os números apontam onde olhar; o `report.html` traz os pares para confirmar de
ouvido.

## Opções úteis

| Opção | Padrão | Para quê |
|---|---|---|
| `--max-per-speaker` | 30 | Clipes sorteados por locutor (0 = todos). 30 já estima bem o teto |
| `--only-group` | – | Analisa só alguns corpora (`--only-group podcast,cv_pt`) |
| `--device` | auto | `cpu` para não usar a GPU |
| `--max-seconds` | 8 | Usa só o trecho central do clipe (mais rápido) |
| `--listen` | 25 | Quantos piores locutores ganham exemplos de áudio no relatório |
| `--intra-warn`, `--intra-bad`, `--dup-threshold`, ... | ver `--help` | Limiares calibrados para ECAPA |

## Onde entra no pipeline de dataset

```
áudio bruto → diarização → corte em clipes → transcrição → PUREZA DE LOCUTOR → seleção/treino
```

Rode depois de montar os clipes. Use `exclude_speakers.txt` / `exclude_clips.txt` para limpar a seleção, e
rode de novo após trocar o diarizador para comparar o teto do corpus antes × depois.
