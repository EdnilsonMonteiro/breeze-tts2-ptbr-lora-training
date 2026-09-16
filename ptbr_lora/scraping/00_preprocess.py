"""Fase 2 - preprocessamento: 48k estereo -> 16k mono PCM16 p/ pyannote e Silero VAD."""
import json
import re
import sys
import time
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
import paths  # noqa: E402

ROOT = paths.ARTIFACTS
DATA = ROOT / "dataScrapping" / "data"
WORK = ROOT / "dataScrapping" / "work"
(WORK / "16k").mkdir(parents=True, exist_ok=True)

PILOT_IDS = None  # escala: processa todos os wavs de data/
TARGET_SR = 16000


def tag_of(name: str) -> str:
    m = re.search(r"\[([^\]]+)\]\.wav$", name)
    return m.group(1) if m else re.sub(r"\W+", "_", name)[:20]


episodes = []
files = sorted(DATA.glob("*.wav"))
print(f"[pre] {len(files)} wavs brutos; piloto: {PILOT_IDS}")
t_all = time.time()
for f in files:
    tid = tag_of(f.name)
    if PILOT_IDS is not None and tid not in PILOT_IDS:
        continue
    info = sf.info(f)
    out = WORK / "16k" / f"{tid}.wav"
    if not out.exists():
        t0 = time.time()
        wav, sr = sf.read(f, dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        g = gcd(sr, TARGET_SR)
        wav16 = resample_poly(wav, TARGET_SR // g, sr // g).astype(np.float32)
        sf.write(out, wav16, TARGET_SR, subtype="PCM_16")
        print(f"[pre] {tid}: {info.samplerate}->{TARGET_SR}Hz {info.channels}ch->mono "
              f"{info.frames / info.samplerate / 60:.1f}min em {time.time() - t0:.0f}s")
    else:
        print(f"[pre] {tid}: intermediario ja existe")
    episodes.append({"tag": tid, "src": str(f), "src_sr": info.samplerate,
                     "dur_h": round(info.frames / info.samplerate / 3600, 3),
                     "w16k": str(out)})

(WORK / "episodes.json").write_text(json.dumps(episodes, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"[pre] OK {len(episodes)} episodios em {time.time() - t_all:.0f}s -> episodes.json")
