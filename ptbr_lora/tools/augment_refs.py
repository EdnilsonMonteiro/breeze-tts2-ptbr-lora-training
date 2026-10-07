"""augment_refs.py — tokens do codec de versoes DEGRADADAS dos clipes de treino (so como referencia).

Para cada clipe de treino que pode servir de referencia (locutor atribuido, 2,5-10,2 s) gera K variantes
(ruido, EQ/banda, reverb, codec; ver core/ref_aug.py), codifica no codec congelado do Breeze e grava
`<training>/tokens_aug/<idx>__a<k>.npz` (mesmo formato dos tokens normais). O treino
(`--ref-aug-p 0.5`) troca, com essa probabilidade, os tokens da REFERENCIA pelos de uma variante;
o ALVO nunca e aumentado. Retomavel (pula o que ja existe). ~15 clipes/s na 4060 Ti.

Uso (Windows, GPU livre):
  set PTBR_ARTIFACTS=<pasta de artefatos>
  set BREEZE_TRAINING_DIR=<pasta de artefatos>\\training_v6
  python ptbr_lora\\tools\\augment_refs.py --k 2 --preview 16
Os wavs de --preview ficam em tokens_aug/_preview/ (original + variante) para voce OUVIR antes de treinar.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_CORE = Path(__file__).resolve().parents[1] / "core"
for _p in (str(_CORE), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import common_breeze as CB  # noqa: E402
import prepare_dataset as PD  # noqa: E402
import ref_aug as RA  # noqa: E402
import refs as R  # noqa: E402


def _codes_ok(path: Path) -> bool:
    try:
        c = np.load(path)["codes"]
        return c.ndim == 2 and c.shape[1] == 16 and c.shape[0] > 0
    except Exception:  # noqa: BLE001
        return False


def _repair(out: Path, n_check: int = 40) -> None:
    """Retomada segura: apaga restos .tmp e os N arquivos mais recentes que estiverem corrompidos
    (processo morto no meio da gravacao)."""
    for t in out.glob("*.tmp.npz"):
        t.unlink(missing_ok=True)
    recent = sorted((p for p in out.glob("*__a*.npz")), key=lambda p: p.stat().st_mtime, reverse=True)[:n_check]
    bad = [p for p in recent if not _codes_ok(p)]
    for p in bad:
        p.unlink(missing_ok=True)
    if bad:
        print(f"[aug] retomada: {len(bad)} arquivo(s) corrompido(s) removido(s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Tokens de referencias degradadas (augmentation).")
    ap.add_argument("--split", default="train")
    ap.add_argument("--k", type=int, default=2, help="variantes por clipe")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=0, help="so os N primeiros (0 = todos)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--preview", type=int, default=16, help="grava N pares original/variante p/ audicao")
    ap.add_argument("--log", type=str, default=None, help="arquivo de log (uma linha a cada 200 itens)")
    ap.add_argument("--p", type=str, default=None,
                    help='JSON com a probabilidade de cada estagio, ex.: {"reverb":0.3,"noise":0.7}')
    args = ap.parse_args()

    recs = {r["idx"]: r for r in PD.load_meta()}
    idxs = [i for i in PD.load_split_file(args.split)
            if i in recs and R.eligible_ref_speaker(recs[i].get("speaker")) and R.ref_duration_ok(recs[i])]
    if args.limit:
        idxs = idxs[: args.limit]
    out = CB.TRAINING / "tokens_aug"
    out.mkdir(parents=True, exist_ok=True)
    _repair(out)
    prev = out / "_preview"
    prev.mkdir(exist_ok=True)
    pj = json.loads(args.p) if args.p else None
    todo = [(i, k) for i in idxs for k in range(args.k) if not (out / f"{i}__a{k}.npz").exists()]
    print(f"[aug] split={args.split} clipes-referencia={len(idxs)} variantes/clipe={args.k} "
          f"a_gerar={len(todo)} -> {out}", flush=True)
    if not todo:
        print("[aug] nada a fazer")
        return

    Qwen3TTSTokenizer = CB.import_qwen_tts()
    atok = Qwen3TTSTokenizer.from_pretrained(str(CB.CKPT / "audio_tokenizer"), device_map=args.device)
    man = (out / "aug_manifest.jsonl").open("a", encoding="utf-8")
    t0, ok, err, n_prev = time.time(), 0, 0, 0
    live = sys.stdout.isatty()                      # terminal real: barra de progresso no lugar
    last_draw = 0.0
    for n, (idx, k) in enumerate(todo):
        try:
            wav, sr = sf.read(str(CB.WAVS24_DIR / f"{idx}.wav"), dtype="float32", always_2d=True)
            wav = wav.mean(axis=1)
            assert sr == CB.SR, f"sr={sr}"
            aug, params = RA.augment(wav, RA.rng_for(args.seed, idx, k), pj)
            enc = atok.encode(aug, sr=CB.SR)
            codes = enc["audio_codes"][0]
            codes = codes.detach().cpu().numpy().astype("int16") if hasattr(codes, "cpu") else codes
            assert codes.ndim == 2 and codes.shape[1] == 16, codes.shape
            assert codes.min() >= 0 and codes.max() < 2051, (codes.min(), codes.max())
            tmp = out / f"{idx}__a{k}.tmp.npz"
            np.savez_compressed(tmp, codes=codes)
            os.replace(tmp, out / f"{idx}__a{k}.npz")                # gravacao atomica
            f0 = int(np.load(CB.TOKENS_DIR / f"{idx}.npz")["codes"].shape[0])
            man.write(json.dumps({"idx": idx, "k": k, "frames": int(codes.shape[0]), "frames_orig": f0,
                                  "params": params}, ensure_ascii=False) + "\n")
            if n_prev < args.preview and k == 0:
                sf.write(prev / f"{idx}__orig.wav", wav, CB.SR, subtype="PCM_16")
                sf.write(prev / f"{idx}__a{k}.wav", aug, CB.SR, subtype="PCM_16")
                n_prev += 1
            ok += 1
        except Exception as exc:  # noqa: BLE001
            err += 1
            print(f"\n[aug] ERRO {idx} a{k}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
        now = time.time()
        done_n = n + 1
        if (done_n % 200 == 0) or done_n == len(todo):
            man.flush()
            if args.log:
                with open(args.log, "a", encoding="utf-8") as lf:
                    lf.write(f"[aug] {done_n}/{len(todo)} ok={ok} err={err}\n")
        if live and (now - last_draw >= 1.0 or done_n == len(todo)):       # linha unica, atualizada no lugar
            last_draw = now
            el = now - t0
            rate = done_n / max(el, 1e-9)
            eta = (len(todo) - done_n) / max(rate, 1e-9)
            w = 30
            fill = int(w * done_n / len(todo))
            sys.stdout.write(f"\r[{'#' * fill}{'-' * (w - fill)}] {done_n}/{len(todo)} "
                             f"({100 * done_n / len(todo):.1f}%) {rate:.1f}/s ok={ok} err={err} "
                             f"decorrido={el / 60:.1f}min restam~{eta / 60:.0f}min   ")
            sys.stdout.flush()
        elif not live and done_n % 200 == 0:
            rate = done_n / max(now - t0, 1e-9)
            print(f"[aug] {done_n}/{len(todo)} ok={ok} err={err} {rate:.1f}/s "
                  f"eta={(len(todo) - done_n) / max(rate, 1e-9) / 60:.1f}min", flush=True)
    man.close()
    print()
    print(f"[aug] FIM ok={ok} err={err} em {(time.time() - t0) / 60:.1f} min | previews: {prev}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
