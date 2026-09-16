"""eval_wer.py — WER/CER automatico das amostras geradas por checkpoint.

Usa faster-whisper (CPU int8 por padrao, para nao competir com o treino na GPU).
Importavel: `evaluate_dir(sample_dir, refs)`.
CLI: python eval_wer.py --dir training/runs/<run>/samples/checkpoint-epoch0
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

_PROJ = Path(__file__).resolve().parents[2]
_CORE = _PROJ / "ptbr_lora" / "core"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

_MODEL = None
_MODEL_KEY: tuple | None = None


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").lower()
    s = re.sub(r"[^\wáàâãéèêíïóôõöúçñ\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _edit(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def wer_cer(ref: str, hyp: str) -> tuple[float, float]:
    r, h = _norm(ref), _norm(hyp)
    rw, hw = r.split(), h.split()
    wer = _edit(rw, hw) / max(1, len(rw))
    cer = _edit(list(r.replace(" ", "")), list(h.replace(" ", ""))) / max(1, len(r.replace(" ", "")))
    return wer, cer


def _get_model(size: str, device: str, compute_type: str):
    global _MODEL, _MODEL_KEY
    key = (size, device, compute_type)
    if _MODEL is None or _MODEL_KEY != key:
        from faster_whisper import WhisperModel

        _MODEL = WhisperModel(size, device=device, compute_type=compute_type)
        _MODEL_KEY = key
    return _MODEL


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower())[:40]


def evaluate_dir(sample_dir: Path, refs: list[tuple[str, str]],
                 size: str = "large-v3", device: str = "cpu",
                 compute_type: str = "int8") -> list[dict]:
    sample_dir = Path(sample_dir)
    wavs = sorted(sample_dir.glob("*.wav"))
    if not wavs:
        return []
    model = _get_model(size, device, compute_type)
    out: list[dict] = []
    for name, text in refs:
        slug = _slug(name)
        match = next((w for w in wavs if w.stem.endswith(slug)), None)
        if match is None:
            continue
        segments, _ = model.transcribe(str(match), language="pt", beam_size=1)
        hyp = " ".join(s.text for s in segments).strip()
        w, c = wer_cer(text, hyp)
        out.append({"name": name, "ref": text, "hyp": hyp,
                    "wer": round(w, 4), "cer": round(c, 4)})
    return out


def main() -> None:
    from train_lora import SAMPLE_TEXTS

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--size", default="large-v3")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8")
    args = ap.parse_args()
    res = evaluate_dir(Path(args.dir), SAMPLE_TEXTS, args.size, args.device, args.compute_type)
    for r in res:
        print(f"{r['name']:>16}  WER={r['wer']:.3f} CER={r['cer']:.3f}  hyp={r['hyp'][:80]}")
    if res:
        import numpy as np

        print(f"MEDIA WER={np.mean([r['wer'] for r in res]):.4f} "
              f"CER={np.mean([r['cer'] for r in res]):.4f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
