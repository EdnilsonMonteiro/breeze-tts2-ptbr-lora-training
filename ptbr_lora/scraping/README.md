# dataScrapping — Pipeline de Dataset de Podcasts PT-BR p/ TTS

Transforma episódios brutos de podcast (YouTube, 48 kHz estéreo, 1–3 h) em
**chunks de treino 24 kHz mono PCM16 + transcrições** no formato do projeto
(`datasets/podcast/texts.csv` → consumido por `prepare_dataset.py`).

## Esteira

```
data/*.wav (48kHz estéreo)
   │  00_preprocess.py      mix mono + reamostragem 16 kHz PCM16 (p/ modelos de análise)
   ▼
work/16k/{tag}.wav ────────► work/episodes.json (manifesto)
   │  01_diarize.py         pyannote segmentation-3.0 (fala vs overlap)
   │                        + SpeechBrain ECAPA (identidade) + clustering 2 passes
   ▼
work/rttm/{tag}.rttm       turnos por locutor
work/diar/{tag}.json       intervalos LIMPOS (sem overlap) por locutor + horas
   │  02_slice.py           Silero VAD ∩ diarização → chunks 1,4–10 s
   │                        pad 175 ms, fala ≥70%, corte do 48k original → 24 kHz
   ▼
datasets/podcast/wavs/{tag}_{N}.wav
work/chunks_index.jsonl    {chunk, tag, speaker, dur, speech_ratio}
   │  03_transcribe.py      API X.ai (4 workers, resumível, retry c/ backoff)
   │                        QC: language=pt, 6–25 chars/s, ≤30% dígitos, NFC,
   │                        limpeza de pontuação, números → extenso (num2words)
   ▼
datasets/podcast/texts.csv (formato wavs/{chunk}.wav==texto)
   │  04_qc_audit.py        relatório + 30 pares sorteados p/ audição humana
   ▼
pilot_audit/playlist.txt   GATE: escute antes de escalar/usar
   05_report_diar.py       tabela de horas por locutor/episódio (uso pontual)
   api_check.py            health-check da API de transcrição
```

## Pré-requisitos

1. Venv do projeto: `C:\IA\Breeze-tts\breeze-tts\venv\Scripts\python.exe`
2. `.env` na raiz do projeto com:
   - `XAI_TRANSCRIBE_KEY` — chave da API de transcrição (https://console.x.ai)
   - `HF_TOKEN` — token HuggingFace com termos **aceitos** em:
     https://hf.co/pyannote/segmentation-3.0
3. Deps instaladas no venv: `pyannote.audio silero-vad num2words speechbrain`
   (instaladas; só reinstalar em ambiente novo)

## Uso

```powershell
$py = "C:\IA\Breeze-tts\breeze-tts\venv\Scripts\python.exe"
# 1) coloque os .wav brutos em data\  (nomes ... [youtubeid].wav; tag = youtubeid)
& $py 00_preprocess.py     # ~6 s/hora de áudio; resumível (pula 16k existente)
& $py 01_diarize.py        # ~40 s/episódio em GPU; escreve work\diar\
& $py 02_slice.py          # ~2 min/h de áudio; resumível (pula tag já sliceada)
& $py 03_transcribe.py     # ~550 chunks/min; resumível via work\transcribe_state.jsonl
& $py 04_qc_audit.py       # relatório + pacote de audição
& $py api_check.py         # (opcional) testa a API antes de rodadas grandes
```

Tudo roda no diretório `dataScrapping`. Interrompeu? Rode o mesmo script de novo —
cada fase é **idempotente/resumível** (tokens `.npz` existentes, tags já fatiadas e
chunks já transcritos são pulados).

## Números da execução real (ago/2026)

| Etapa | Valor |
|---|---|
| Episódios brutos | 18 (36 h, 23 GB, 48 kHz estéreo) |
| Fala exclusiva pós-diarização | 26,3 h |
| Chunks fatiados | 23.134 (22,4 h novos + piloto) |
| Aceitos na transcrição | **22.228 / 96,1% — 23,2 h** |
| Tempo total da esteira | ~2,5 h (diarização ~12 min GPU; transcrição ~40 min) |
| Rejeições | 3,9% — densidade textual extrema (<6 ou >25 chars/s) ou vazio |

Rejeições por densidade alta (25–30 c/s) são fala real e rápida; se quiser
recuperá-las, suba `CHARS_PER_S` no `03_transcribe.py`, remova as linhas
correspondentes de `work\transcribe_state.jsonl` e rode o 03 de novo.

## Formatos

- `work/chunks_index.jsonl` — `{"chunk","tag","speaker","dur","speech_ratio","spans"}`
  (`speaker` NÃO compartilha identidade entre episódios — cada tag tem SPEAKER_00… próprios)
- `datasets/podcast/texts.csv` — linhas `wavs/{chunk}.wav==texto` (parser do
  `prepare_dataset.py` aceita direto; nomes `podcast_*` não colidem com `sample-*` da Tata)
- `work/transcribe_state.jsonl` — manifesto da API:
  `{"chunk","status":"ok|rejected","text"|"reason"}`

## Decisões técnicas

- **Diarização sem pipeline pronto**: o `pyannote.audio` 4.x resolve
  `speaker-diarization-3.1` → repo gated *community-1* (mais um consentimento), e o
  3.3.2 (stack clássica) quebra com `torchaudio>=2.9`. Usamos a receita mínima oficial:
  `segmentation-3.0` (ativ. >0.6, frames exclusivos = sem overlap) + embeddings ECAPA +
  clustering aglomerativo (limiar 0.45) com dissolução de clusters < 60 s.
- **Chunks saem do 48k original** (não do intermediário 16k) — fidelidade máxima na
  reamostragem para 24 kHz.
- **Sem denoise**: fala de estúdio é limpa; VAD + diarização já descartam música
  e fala cruzada. (DeepFilterNet foi descartado: build Rust falha no Windows.)
- **Números → extenso** (`num2words` pt-BR): o corpus Tata é 99,8% sem dígitos —
  consistência com o pré-treino.
- **Disco**: os intermediários `work\16k\` (~4 GB) podem ser apagados após a fase 2;
  a esteira completa consumiu ~11 GB (chunks + tokens + wavs24). Monitore espaço antes
  de rodadas grandes (um episódio de 3 h gera ~2 GB de chunk wavs).

## Integração com o treino

Ver `..\PLANO-DATASET-PODCAST.md` e `..\tutorial-lora\`:
`prepare_dataset.py process` (com `BREEZE_DATASET_DIR=…\datasets\podcast`) tokeniza os
chunks, `finalize` faz o split estratificado por corpus com campo `speaker`, e o
treino usa o variant `ref_edit_auto` (referência de outro clipe do MESMO locutor).
