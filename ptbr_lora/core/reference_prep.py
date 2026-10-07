"""reference_prep.py — prepara o audio de referencia (prompt) no formato do TREINO.

No treino o audio de referencia entra no codec como esta no corpus (24 kHz, clipes de
~3-10 s). A UI antiga reamostrava tudo para 48 kHz (`reference_48k`) — distribuicao
diferente da vista no treino. Aqui:

  mode="train": mono -> trim (top_db=40) -> resample 24 kHz -> peak-norm 0.95 se pico
                <0.30 ou >0.99  (default; e o que o adapter viu)
  mode="48k"  : comportamento anterior da UI (mono, 48 kHz, sem trim)
  mode="raw"  : usa o arquivo original

Devolve o caminho de um WAV cacheado (hash de caminho+mtime+modo). Deps pesadas
(librosa/soundfile) sao importadas dentro da funcao.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

TRAIN_SR = 24_000
MAX_REF_S = 10.2          # teto de duracao do clipe no treino
IDEAL_REF_S = (3.0, 10.0)
PEAK_NORM = 0.95

MODES = ("train", "48k", "raw")


def peak_norm(wav, lo: float = 0.30, hi: float = 0.99, target: float = PEAK_NORM):
    import numpy as np

    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    if peak < lo or peak > hi:
        wav = wav * (target / max(peak, 1e-9))
    return np.clip(wav, -1.0, 1.0)


def duration_warning(seconds: float) -> str | None:
    if seconds > MAX_REF_S:
        return (f"Referência com {seconds:.1f}s: o treino só viu clipes de até ~10 s — "
                f"use um trecho de {IDEAL_REF_S[0]:.0f}–{IDEAL_REF_S[1]:.0f} s.")
    if seconds < 2.0:
        return f"Referência muito curta ({seconds:.1f}s): use {IDEAL_REF_S[0]:.0f}–{IDEAL_REF_S[1]:.0f} s."
    return None


def prepare_reference(wav_path, cache_dir, mode: str = "train") -> tuple[Path, float]:
    """Retorna (wav_preparado, duracao_s). Nao altera o original."""
    import librosa
    import numpy as np
    import soundfile as sf

    if mode not in MODES:
        raise ValueError(f"mode invalido: {mode!r} (use {MODES})")
    src = Path(wav_path)
    if mode == "raw":
        info = sf.info(str(src))
        return src, float(info.frames) / float(info.samplerate)

    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{src.resolve()}|{src.stat().st_mtime_ns}|{mode}".encode()).hexdigest()[:16]
    out = d / f"ref_{mode}_{key}.wav"
    sr_out = TRAIN_SR if mode == "train" else 48_000
    if out.is_file():
        info = sf.info(str(out))
        return out, float(info.frames) / float(info.samplerate)

    wav, sr = sf.read(str(src), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if mode == "train":
        wav, _ = librosa.effects.trim(wav, top_db=40)
    if sr != sr_out:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=sr_out)
    if mode == "train":
        wav = peak_norm(wav)
    wav = np.clip(wav, -1.0, 1.0).astype("float32")
    sf.write(str(out), wav, sr_out, subtype="PCM_16")
    return out, len(wav) / float(sr_out)
