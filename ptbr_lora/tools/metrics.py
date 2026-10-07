"""metrics.py — metricas objetivas para comparar geracoes de voz.

Eixos:
- Identidade  : cos de embedding ECAPA vs uma/mais referencias.
- Inteligibilidade: WER/CER (faster-whisper) vs texto-alvo.
- Prosodia    : duracao, F0 (media/std), energia (RMS).

Uso como modulo (import metrics) ou CLI rapida para um unico WAV.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_CORE = Path(__file__).resolve().parents[1] / "core"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))
import asr_metrics as AM  # noqa: E402
import paths  # noqa: E402

_ECAPA = None
_ASR = None
_ASR_KEY: tuple | None = None


# ----------------------------------------------------------------- identidade
def load_ecapa(device: str = "cpu"):
    global _ECAPA
    if _ECAPA is None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except Exception:  # noqa: BLE001
            from speechbrain.pretrained import EncoderClassifier
        _ECAPA = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(paths.SCRAPING_WORK / "spkrec-ecapa"),
            run_opts={"device": device},
        )
    return _ECAPA


def embed(path, device: str = "cpu") -> np.ndarray:
    import librosa
    import torch

    clf = load_ecapa(device)
    wav, _sr = librosa.load(str(path), sr=16000, mono=True)
    x = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        e = clf.encode_batch(x).squeeze()
    e = e / (e.norm() + 1e-9)
    return e.detach().cpu().numpy().astype(np.float32)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def embed_array(y: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Embedding ECAPA de um array mono 16 kHz."""
    import torch

    clf = load_ecapa(device)
    x = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        e = clf.encode_batch(x).squeeze()
    e = e / (e.norm() + 1e-9)
    return e.detach().cpu().numpy().astype(np.float32)


def temporal_cos(path, ref_emb: np.ndarray, device: str = "cpu",
                 seg_s: float = 6.0) -> tuple[float, float, float]:
    """cos(ref, inicio) , cos(ref, fim) e deriva (inicio - fim).

    Mede a observacao "comeca parecido e depois perde a referencia": deriva
    positiva grande = a identidade cai ao longo do audio.
    """
    import librosa

    y, _ = librosa.load(str(path), sr=16000, mono=True)
    n = int(seg_s * 16000)
    if len(y) < 2 * n:
        m = max(1, len(y) // 2)
        first, last = y[:m], y[-m:]
    else:
        first, last = y[:n], y[-n:]
    c_first = cos(ref_emb, embed_array(first, device))
    c_last = cos(ref_emb, embed_array(last, device))
    return c_first, c_last, c_first - c_last


# ----------------------------------------------------------- inteligibilidade
def load_asr(size: str = "large-v3", device: str = "cpu", compute_type: str = "int8"):
    global _ASR, _ASR_KEY
    key = (size, device, compute_type)
    if _ASR is None or _ASR_KEY != key:
        from faster_whisper import WhisperModel

        _ASR = WhisperModel(size, device=device, compute_type=compute_type)
        _ASR_KEY = key
    return _ASR


def transcribe(path, size: str = "large-v3", device: str = "cpu",
               compute_type: str = "int8", lang: str = "pt") -> str:
    """Sem VAD e sem condition_on_previous_text: o VAD cortava cauda/arrasto e escondia falhas."""
    model = load_asr(size, device, compute_type)
    segs, _ = model.transcribe(str(path), language=lang, beam_size=5, vad_filter=False,
                               condition_on_previous_text=False)
    return " ".join(s.text for s in segs).strip()


def _norm(s: str) -> str:
    return AM.normalize_for_wer(s, "pt", True)


def wer_cer(ref: str, hyp: str, lang: str = "pt", normalize: bool = True) -> tuple[float, float]:
    """WER/CER com normalizacao (text_norm) nos DOIS lados — implementacao unica em asr_metrics."""
    return AM.wer_cer(ref, hyp, lang=lang, normalize=normalize)


def word_hits(ref: str, hyp: str, words: list[str]) -> dict[str, bool]:
    """Diz se cada palavra-alvo aparece corretamente na hipotese (ASR)."""
    hn = _norm(hyp)
    out = {}
    for w in words:
        out[w] = _norm(w) in hn
    return out


# ------------------------------------------------------------------- prosodia
def prosody(path, sr: int = 24000) -> dict:
    import librosa

    y, sr = librosa.load(str(path), sr=sr, mono=True)
    dur = len(y) / sr
    try:
        f0 = librosa.yin(y, fmin=60, fmax=400, sr=sr)
        f0 = f0[np.isfinite(f0)]
    except Exception:  # noqa: BLE001
        f0 = np.array([])
    rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
    return {
        "dur": round(float(dur), 3),
        "f0_mean": round(float(np.mean(f0)), 2) if len(f0) else 0.0,
        "f0_std": round(float(np.std(f0)), 2) if len(f0) else 0.0,
        "rms": round(rms, 4),
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.ref:
        print("cos(ref) =", round(cos(embed(a.ref, a.device), embed(a.wav, a.device)), 3))
    print("prosody =", prosody(a.wav))
