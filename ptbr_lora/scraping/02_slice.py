"""Fase 4 - slicing: Silero VAD ∩ diarizacao -> chunks 3-10s @24k mono PCM16."""
import json
import sys
import time
from pathlib import Path

import soundfile as sf
from scipy.signal import resample_poly
from silero_vad import load_silero_vad, get_speech_timestamps

sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
import paths  # noqa: E402

ROOT = paths.ARTIFACTS
WORK = ROOT / "dataScrapping" / "work"
OUT_WAVS = ROOT / "datasets" / "podcast" / "wavs"
OUT_WAVS.mkdir(parents=True, exist_ok=True)

SR16, SR24, SR48 = 16000, 24000, 48000
MIN_GAP_MS = 300          # agrupa segmentos do mesmo locutor se gap menor que isso
MAX_CORE_S = 9.5          # nucleo maximo (9.5 + 0.35 pad = 9.85 < limite 10.2 do treino)
MIN_CHUNK_S = 1.0
PAD_S = 0.175             # padding alvo do usuario (150-200 ms)
SPEECH_RATIO_MIN = 0.70
MIN_SPEECH_S = 0.40       # minimo de span VAD limpo p/ considerar

episodes = json.loads((WORK / "episodes.json").read_text(encoding="utf-8"))
# resume: pula episodios ja sliceados (piloto) e continua numeracao por tag
done_tags, done_count = {}, {}
if (WORK / "chunks_index.jsonl").exists():
    for ln in (WORK / "chunks_index.jsonl").read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            done_tags[r["tag"]] = True
            done_count[r["tag"]] = max(done_count.get(r["tag"], 0),
                                       int(r["chunk"].rsplit("_", 1)[1].split(".")[0]))
model = load_silero_vad()
index_f = (WORK / "chunks_index.jsonl").open("a", encoding="utf-8")
stats = {"chunks": 0, "h": 0.0}
t_all = time.time()

for ep in episodes:
    tag = ep["tag"]
    if tag in done_tags:
        print(f"[slice] {tag}: ja sliceado, pulando", flush=True)
        continue
    diar_f = WORK / "diar" / f"{tag}.json"
    if not diar_f.exists():
        print(f"[slice] {tag}: sem diarizacao, pulando", flush=True)
        continue
    t0 = time.time()
    wav16, _ = sf.read(ep["w16k"], dtype="float32")
    vad = get_speech_timestamps(wav16, model, sampling_rate=SR16,
                                threshold=0.5, min_speech_duration_ms=250,
                                min_silence_duration_ms=100, speech_pad_ms=0)
    vad_spans = [(v["start"] / SR16, v["end"] / SR16) for v in vad]

    dia = json.loads(diar_f.read_text(encoding="utf-8"))
    n = done_count.get(tag, 0)
    for spk, clean in sorted(dia["clean"].items()):
        # VAD ∩ intervalos limpos do locutor
        spans = []
        for cs, ce in clean:
            for vs, ve in vad_spans:
                a, b = max(cs, vs), min(ce, ve)
                if b - a >= MIN_SPEECH_S:
                    spans.append((a, b))
        if not spans:
            continue
        spans.sort()
        # agrupa mesmo locutor com gap pequeno; corta nucleo em MAX_CORE_S
        groups, cur = [], [spans[0]]
        for a, b in spans[1:]:
            pa, pb = cur[-1]
            if a - pb < MIN_GAP_MS / 1000 and (b - cur[0][0]) <= MAX_CORE_S:
                cur.append((a, b))
            else:
                groups.append(cur)
                cur = [(a, b)]
        groups.append(cur)
        for g in groups:
            a, b = g[0][0], g[-1][1]
            if b - a < MIN_CHUNK_S:
                continue
            # padding e recorte final
            a2, b2 = max(0.0, a - PAD_S), b + PAD_S
            dur = b2 - a2
            if not (MIN_CHUNK_S <= dur <= 10.0):
                continue
            # ratio de fala VAD dentro do chunk
            sp = sum(min(ve, b2) - max(vs, a2) for vs, ve in vad_spans if ve > a2 and vs < b2)
            if sp / dur < SPEECH_RATIO_MIN:
                continue
            # exporta do original 48k estereo -> 24k mono
            i0, i1 = int(a2 * SR48), int(b2 * SR48)
            try:
                seg, _ = sf.read(ep["src"], start=i0, frames=i1 - i0, dtype="float32", always_2d=True)
            except Exception as ex:
                print(f"[slice] erro leitura {tag}@{a2:.1f}: {ex}", flush=True)
                continue
            seg = seg.mean(axis=1)
            seg24 = resample_poly(seg, SR24, SR48)
            n += 1
            name = f"{tag}_{n:05d}.wav"
            sf.write(OUT_WAVS / name, seg24, SR24, subtype="PCM_16")
            index_f.write(json.dumps({"chunk": name, "tag": tag, "speaker": spk,
                                      "dur": round(dur, 2), "speech_ratio": round(sp / dur, 2),
                                      "spans": [[round(x, 2) for x in s] for s in g]},
                                     ensure_ascii=False) + "\n")
            stats["chunks"] += 1
            stats["h"] += dur / 3600
    print(f"[slice] {tag}: {n} chunks em {time.time() - t0:.0f}s", flush=True)

index_f.close()
print(f"[slice] OK {stats['chunks']} chunks / {stats['h']:.2f}h em {time.time() - t_all:.0f}s "
      f"-> chunks_index.jsonl", flush=True)
