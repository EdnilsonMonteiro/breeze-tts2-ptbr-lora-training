"""ref_aug.py — augmentation do AUDIO DE REFERENCIA (prompt) para o treino de clonagem. So numpy.

Por que: no treino a referencia e outro clipe do MESMO locutor, quase sempre da mesma sala/mic/
canal que o alvo -> o modelo pode "copiar o canal" em vez de aprender timbre. Com a referencia
degradada de formas aleatorias (ruido, EQ/banda, reverb, codec), o alvo continua limpo e o unico
sinal estavel entre os dois e a IDENTIDADE da voz. O alvo NUNCA e aumentado.

Todas as operacoes preservam o comprimento (mesmo n.o de frames do codec). Deterministico por
(seed, idx, k). Cadeia: reverb -> EQ/banda -> codec (mu-law) -> ruido -> normaliza pico.
"""
from __future__ import annotations

import hashlib

import numpy as np

SR = 24_000

# probabilidade de cada estagio (a cadeia sempre aplica >= 1 estagio)
DEFAULT_P = {"reverb": 0.35, "eq": 0.55, "codec": 0.30, "noise": 0.65}


def rng_for(seed: int, idx: str, k: int) -> np.random.Generator:
    h = hashlib.sha1(f"{seed}|{idx}|{k}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if len(x) else 0.0


def colored_noise(n: int, color: str, rng: np.random.Generator) -> np.ndarray:
    """white | pink (1/f) | brown (1/f^2); RMS = 1."""
    w = rng.standard_normal(n)
    if color == "white":
        out = w
    else:
        spec = np.fft.rfft(w)
        f = np.fft.rfftfreq(n, 1.0 / SR)
        f[0] = f[1] if len(f) > 1 else 1.0
        alpha = 1.0 if color == "pink" else 2.0
        spec = spec / (f ** (alpha / 2.0))
        out = np.fft.irfft(spec, n)
    r = _rms(out)
    return (out / r) if r > 1e-12 else out


def add_noise(wav: np.ndarray, snr_db: float, color: str, rng: np.random.Generator,
              hum: bool = False) -> np.ndarray:
    n = len(wav)
    sig = _rms(wav)
    if sig < 1e-6:
        return wav.copy()
    noise = colored_noise(n, color, rng)
    if hum:                                           # 50/60 Hz + harmonicas (rede eletrica)
        f0 = float(rng.choice([50.0, 60.0]))
        t = np.arange(n) / SR
        h = sum(np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 2 * np.pi)) / k for k in (1, 2, 3))
        noise = noise + 0.5 * h / max(_rms(h), 1e-9)
        noise = noise / max(_rms(noise), 1e-9)
    gain = sig / (10 ** (snr_db / 20.0))
    return wav + gain * noise


def eq_response(freqs: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Resposta em magnitude: passa-baixa (banda do mic/codec), passa-alta, inclinacao e 1-2 picos."""
    f = np.maximum(freqs, 1.0)
    H = np.ones_like(f)
    p: dict = {}
    if rng.random() < 0.7:                            # banda limitada (laptop/telefone/streaming)
        fc = float(rng.uniform(3400, 9500))
        H *= 1.0 / np.sqrt(1.0 + (f / fc) ** 8)       # Butterworth ordem 4 (magnitude)
        p["lowpass_hz"] = round(fc)
    if rng.random() < 0.5:
        fc = float(rng.uniform(70, 320))
        H *= 1.0 / np.sqrt(1.0 + (fc / f) ** 4)
        p["highpass_hz"] = round(fc)
    if rng.random() < 0.6:                            # inclinacao espectral +-6 dB entre 100 Hz e 8 kHz
        tilt = float(rng.uniform(-6, 6))
        H *= 10 ** ((tilt * np.log2(f / 1000.0) / 3.3) / 20.0)
        p["tilt_db"] = round(tilt, 1)
    for _ in range(int(rng.integers(0, 3))):          # picos/vales (ressonancia de sala/mic)
        fc, g, q = float(rng.uniform(200, 5000)), float(rng.uniform(-6, 6)), float(rng.uniform(0.7, 3.0))
        H *= 10 ** ((g * np.exp(-0.5 * (np.log2(f / fc) * q * 2) ** 2)) / 20.0)
        p.setdefault("peaks", []).append([round(fc), round(g, 1)])
    return H, p


def apply_eq(wav: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    n = len(wav)
    spec = np.fft.rfft(wav.astype(np.float64))
    H, p = eq_response(np.fft.rfftfreq(n, 1.0 / SR), rng)
    out = np.fft.irfft(spec * H, n)
    r_in, r_out = _rms(wav), _rms(out)
    if r_out > 1e-9:
        out = out * (r_in / r_out)
    return out, p


def synthetic_rir(rt60: float, rng: np.random.Generator) -> np.ndarray:
    n = max(16, int(rt60 * SR))
    t = np.arange(n) / SR
    rir = rng.standard_normal(n) * np.exp(-6.9078 * t / rt60)      # -60 dB em rt60
    rir[0] = 0.0
    rir = rir / max(_rms(rir), 1e-9) * float(rng.uniform(0.15, 0.6))   # reverb/direto variavel
    rir[0] = 1.0                                                    # caminho direto
    return rir


def apply_reverb(wav: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    rt60 = float(rng.uniform(0.15, 0.75))
    rir = synthetic_rir(rt60, rng)
    n = len(wav)
    L = n + len(rir) - 1
    nfft = 1 << (L - 1).bit_length()
    out = np.fft.irfft(np.fft.rfft(wav.astype(np.float64), nfft) * np.fft.rfft(rir, nfft), nfft)[:n]
    r_in, r_out = _rms(wav), _rms(out)
    if r_out > 1e-9:
        out = out * (r_in / r_out)
    return out, {"rt60": round(rt60, 2)}


def mulaw_roundtrip(wav: np.ndarray, bits: int = 8, mu: float = 255.0) -> np.ndarray:
    x = np.clip(wav, -1.0, 1.0)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = 2 ** (bits - 1) - 1
    y = np.round(y * q) / q
    return np.sign(y) * ((1.0 + mu) ** np.abs(y) - 1.0) / mu


def augment(wav: np.ndarray, rng: np.random.Generator, p: dict | None = None
            ) -> tuple[np.ndarray, dict]:
    """wav float32 mono 24 kHz -> (wav_aug float32 mesmo comprimento, parametros aplicados)."""
    P = dict(DEFAULT_P)
    if p:
        P.update(p)
    x = np.asarray(wav, dtype=np.float64)
    params: dict = {}
    if len(x) < 64 or _rms(x) < 1e-6:
        return np.asarray(wav, dtype=np.float32).copy(), {"skipped": "silencio/curto"}
    chosen = {k: bool(rng.random() < P[k]) for k in ("reverb", "eq", "codec", "noise")}
    if not any(chosen.values()):
        chosen["noise" if rng.random() < 0.5 else "eq"] = True
    if chosen["reverb"]:
        x, params["reverb"] = apply_reverb(x, rng)
    if chosen["eq"]:
        x, params["eq"] = apply_eq(x, rng)
    if chosen["codec"]:
        bits = int(rng.choice([6, 7, 8]))
        pre = _rms(x)
        x = mulaw_roundtrip(x / max(np.max(np.abs(x)), 1e-9) * 0.95, bits=bits)
        r = _rms(x)
        x = x * (pre / r) if r > 1e-9 else x
        params["codec"] = {"mulaw_bits": bits}
    if chosen["noise"]:
        snr = float(rng.uniform(8.0, 30.0))
        color = str(rng.choice(["white", "pink", "brown"]))
        hum = bool(rng.random() < 0.15)
        x = add_noise(x, snr, color, rng, hum=hum)
        params["noise"] = {"snr_db": round(snr, 1), "color": color, "hum": hum}
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak > 0.99 or peak < 0.30:                    # mesma regra do preparo do corpus (PEAK_NORM 0.95)
        x = x * (0.95 / max(peak, 1e-9))
    x = np.clip(x, -1.0, 1.0)
    return x.astype(np.float32), params


def _u(*parts) -> float:
    """Uniforme [0,1) deterministica (mesma construcao de refs._u)."""
    h = hashlib.sha1("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def choose_aug_variant(seed: int, salt: int, rec_idx: str, avail: list[int] | None, p: float) -> int | None:
    """Indice `k` da variante aumentada da REFERENCIA a usar neste sorteio (None = referencia original).

    Deterministico em (seed, salt, rec_idx): `salt` e a posicao do sorteio (modo por massa) ou a epoca.
    Com `p` e variantes disponiveis, escolhe aumentada com prob. `p`, entre as K variantes, uniformemente.
    """
    if p <= 0 or not avail:
        return None
    if _u("aug", seed, salt, rec_idx) >= p:
        return None
    return avail[int(_u("augk", seed, salt, rec_idx) * len(avail)) % len(avail)]

