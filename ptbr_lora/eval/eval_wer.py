"""eval_wer.py — WER/CER automatico das amostras geradas por checkpoint.

Usa faster-whisper (CPU int8 por padrao, para nao competir com o treino na GPU).
Importavel: `evaluate_dir(sample_dir, refs)` e `summarize_results(res)`.
CLI: python eval_wer.py --dir training/runs/<run>/samples/checkpoint-epoch0

Protocolo (corrige a auditoria 2026-09):
  * normalizacao (text_norm) em ref E hyp, ANTES de minusculizar — ligada por padrao;
  * idioma por amostra: nomes `*-en` transcrevem com language="en" e ficam FORA da media pt;
  * decodificacao SEM vad_filter e SEM condition_on_previous_text, beam 5: o VAD cortava
    cauda/silencio e escondia falhas (fala arrastada, alucinacao) — justamente o que se quer ver;
  * alem do WER: S/D/I, duracao, falhas catastroficas (WER>0,5, duracao absurda, repeticao)
    e IC 95 % bootstrap na media.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_PROJ = Path(__file__).resolve().parents[2]
_CORE = _PROJ / "ptbr_lora" / "core"
_TOOLS = _PROJ / "ptbr_lora" / "tools"
for _p in (str(_CORE), str(_TOOLS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import asr_metrics as AM  # noqa: E402

_MODEL = None
_MODEL_KEY: tuple | None = None


def wer_cer(ref: str, hyp: str, do_norm: bool = True, lang: str = "pt") -> tuple[float, float]:
    """Compat com a API antiga (agora normaliza por padrao)."""
    return AM.wer_cer(ref, hyp, lang=lang, normalize=do_norm)


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


def lang_of(name: str) -> str:
    return "en" if name.endswith("-en") else "pt"


def transcribe(model, wav_path, lang: str) -> str:
    segments, _ = model.transcribe(str(wav_path), language=lang, beam_size=5,
                                   vad_filter=False, condition_on_previous_text=False)
    return " ".join(s.text for s in segments).strip()


def evaluate_dir(sample_dir: Path, refs: list[tuple[str, str]],
                 size: str = "large-v3", device: str = "cpu",
                 compute_type: str = "int8", do_norm: bool = True,
                 match: str | None = None) -> list[dict]:
    import soundfile as sf

    sample_dir = Path(sample_dir)
    wavs = sorted(sample_dir.glob("*.wav"))
    if match:
        wavs = [w for w in wavs if match in w.name]
    if not wavs:
        return []
    model = _get_model(size, device, compute_type)
    out: list[dict] = []
    for name, text in refs:
        slug = _slug(name)
        # aceita tanto "00_ola-pt.wav" (samples) quanto "00_ola-pt_final.wav" (reference/)
        wav = next((w for w in wavs if slug in w.stem), None)
        if wav is None:
            continue
        lang = lang_of(name)
        hyp = transcribe(model, wav, lang)
        d = AM.wer_detail(text, hyp, lang=lang, normalize=do_norm)
        info = sf.info(str(wav))
        dur = info.frames / float(info.samplerate)
        n_words = len(AM.normalize_for_wer(text, lang, do_norm).split())
        why = AM.catastrophic(d["wer"], dur, n_words, hyp)
        out.append({"name": name, "lang": lang, "ref": text, "hyp": hyp,
                    "wer": round(d["wer"], 4), "cer": round(d["cer"], 4),
                    "S": d["S"], "D": d["D"], "I": d["I"], "dur_s": round(dur, 2),
                    "fail": why})
    return out


def summarize_results(res: list[dict]) -> dict:
    """{'pt': {...}, 'en': {...}} com media, IC95, mediana, pior caso e nº de falhas."""
    summ: dict = {}
    for lang in ("pt", "en"):
        rows = [r for r in res if r.get("lang", lang_of(r["name"])) == lang]
        if not rows:
            continue
        w = AM.summarize([r["wer"] for r in rows], [bool(r.get("fail")) for r in rows])
        c = AM.summarize([r["cer"] for r in rows])
        summ[lang] = {"n": len(rows), "wer_mean": w["mean"], "wer_ci95": w["ci95"],
                      "wer_median": w["median"], "wer_max": max(r["wer"] for r in rows),
                      "cer_mean": c["mean"], "n_fail": sum(1 for r in rows if r.get("fail")),
                      "fails": [r["name"] for r in rows if r.get("fail")]}
    return summ


def main() -> None:
    from sample_texts import SAMPLE_TEXTS

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--size", default="large-v3")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--no-normalize", action="store_true",
                    help="NAO normaliza numeros/siglas (default: normaliza ref E hyp)")
    ap.add_argument("--match", default=None,
                    help="so considera wavs cujo nome contem esta string (ex.: step4500)")
    args = ap.parse_args()
    res = evaluate_dir(Path(args.dir), SAMPLE_TEXTS, args.size, args.device,
                       args.compute_type, not args.no_normalize, args.match)
    for r in res:
        flag = f" FALHA={','.join(r['fail'])}" if r["fail"] else ""
        print(f"{r['name']:>16} [{r['lang']}] WER={r['wer']:.3f} CER={r['cer']:.3f} "
              f"(S{r['S']}/D{r['D']}/I{r['I']}) {r['dur_s']:.1f}s{flag}  hyp={r['hyp'][:70]}")
    for lang, s in summarize_results(res).items():
        lo, hi = s["wer_ci95"]
        print(f"[{lang}] n={s['n']} WER={s['wer_mean']:.4f} (IC95 {lo:.3f}-{hi:.3f}) "
              f"mediana={s['wer_median']:.3f} max={s['wer_max']:.3f} CER={s['cer_mean']:.4f} "
              f"falhas={s['n_fail']} {s['fails']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
