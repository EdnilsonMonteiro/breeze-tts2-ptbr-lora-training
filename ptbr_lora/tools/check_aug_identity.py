"""check_aug_identity.py — a referencia degradada ainda e a MESMA voz? (SECS por ECAPA, so CPU)

Regenera (deterministico: mesma seed/chain do augment_refs) N variantes aumentadas de clipes de treino e mede
  aug    = cos(original, variante)                      <- o que a augmentation faz com a identidade
  mesma  = cos(original, outro clipe LIMPO da mesma voz)  <- teto realista de "mesma voz"
  outra  = cos(original, clipe de OUTRA voz)             <- piso
Leitura: se `aug` fica perto de `mesma`, a identidade sobrevive; se cai para perto de `outra`, a degradacao
esta pesada demais (baixe --ref-aug-p ou suavize os estagios em augment_refs --p). Mostra tambem por estagio.
Uso: python ptbr_lora\\tools\\check_aug_identity.py [--n 200] [--k 0]
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

_CORE = Path(__file__).resolve().parents[1] / "core"
_TOOLS = Path(__file__).resolve().parent
for p in (str(_CORE), str(_TOOLS)):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import metrics  # noqa: E402
import paths  # noqa: E402
import ref_aug as RA  # noqa: E402
import refs as R  # noqa: E402
import meta_io  # noqa: E402
import splits as SPL  # noqa: E402


def _emb(wav24: np.ndarray) -> np.ndarray:
    import librosa

    y = librosa.resample(wav24.astype(np.float32), orig_sr=24000, target_sr=16000)
    r = float(np.sqrt(np.mean(y ** 2)))
    if r > 1e-6:
        y = y * (0.05 / r)                                   # iguala RMS (ECAPA sensivel ao nivel)
    return metrics.embed_array(y, "cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234, help="mesma seed do augment_refs")
    args = ap.parse_args()
    tr = paths.TRAINING
    recs = {r["idx"]: r for r in meta_io.load_meta_records(tr / "dataset_meta.jsonl")}
    train = [i for i in SPL.load_split_file(tr, "train") if i in recs
             and R.eligible_ref_speaker(recs[i].get("speaker")) and R.ref_duration_ok(recs[i])]
    by_spk = defaultdict(list)
    for i in train:
        by_spk[recs[i]["speaker"]].append(i)
    rnd = random.Random(7)
    idxs = rnd.sample(train, min(args.n, len(train)))
    spks = sorted(by_spk)
    res = defaultdict(list)
    per_stage = defaultdict(list)
    for n, idx in enumerate(idxs):
        wav, sr = sf.read(str(tr / "wavs24" / f"{idx}.wav"), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        aug, params = RA.augment(wav, RA.rng_for(args.seed, idx, args.k), None)
        e0, ea = _emb(wav), _emb(aug)
        res["aug"].append(metrics.cos(e0, ea))
        spk = recs[idx]["speaker"]
        same = [j for j in by_spk[spk] if j != idx]
        if same:
            w2, _ = sf.read(str(tr / "wavs24" / f"{rnd.choice(same)}.wav"), dtype="float32", always_2d=True)
            res["mesma"].append(metrics.cos(e0, _emb(w2.mean(axis=1))))
        other = rnd.choice([s for s in spks if s != spk])
        w3, _ = sf.read(str(tr / "wavs24" / f"{rnd.choice(by_spk[other])}.wav"), dtype="float32", always_2d=True)
        res["outra"].append(metrics.cos(e0, _emb(w3.mean(axis=1))))
        for st in params:
            per_stage[st].append(res["aug"][-1])
        if (n + 1) % 25 == 0:
            print(f"[{n + 1}/{len(idxs)}]", flush=True)

    def q(v):
        v = np.array(v)
        return f"media {v.mean():.3f} | p10 {np.percentile(v, 10):.3f} p50 {np.percentile(v, 50):.3f} p90 {np.percentile(v, 90):.3f}"

    print("\n=== SECS (cos ECAPA) com o clipe original ===")
    for k in ("mesma", "aug", "outra"):
        print(f"  {k:6s}: {q(res[k])}")
    print("\n=== SECS(original, aug) por estagio presente ===")
    for st, v in sorted(per_stage.items()):
        print(f"  {st:7s} n={len(v):4d}: media {np.mean(v):.3f}")
    ceil, floor, a = np.mean(res["mesma"]), np.mean(res["outra"]), np.mean(res["aug"])
    keep = (a - floor) / max(ceil - floor, 1e-6)
    print(f"\nidentidade preservada (0=voz qualquer, 1=outro clipe limpo da mesma voz): {keep:.2f}")
    print("-> OK (>= 0.6)" if keep >= 0.6 else "-> PESADO: reduza --ref-aug-p ou suavize a degradacao")


if __name__ == "__main__":
    main()
