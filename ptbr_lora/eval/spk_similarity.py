"""spk_similarity.py — similaridade de locutor (ECAPA) entre referência e áudios gerados.

Mede a identidade de voz: para cada áudio gerado, calcula o cosseno do embedding
ECAPA contra cada referência. Regra prática: ~0,7+ = mesmo locutor; <0,3 = outra pessoa.

Uso:
  python spk_similarity.py --dir <pasta> [--refs a.wav b.wav] [--gens c.wav d.wav]

Sem `--refs`/`--gens`, descobre automaticamente em `--dir`:
  referências = `ref_*.wav`; gerações = `*.wav` (exceto as referências).
Default de `--dir`: <PTBR_ARTIFACTS>/training/clone_out
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_CORE = Path(__file__).resolve().parents[1] / "core"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))
import paths  # noqa: E402


def _embed(clf, path: Path, device: str) -> torch.Tensor:
    import librosa

    wav, _sr = librosa.load(str(path), sr=16000, mono=True)
    x = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        e = clf.encode_batch(x).squeeze()
    return (e / (e.norm() + 1e-9)).float()


def _encoder(device: str, savedir: str):
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except Exception:  # noqa: BLE001
        from speechbrain.pretrained import EncoderClassifier

    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=savedir,
        run_opts={"device": device},
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Similaridade de locutor (ECAPA).")
    ap.add_argument("--dir", default=str(paths.TRAINING / "clone_out"),
                    help="pasta com referências e gerações")
    ap.add_argument("--refs", nargs="*", default=None,
                    help="arquivos de referência (default: ref_*.wav em --dir)")
    ap.add_argument("--gens", nargs="*", default=None,
                    help="áudios gerados (default: *.wav em --dir, exceto as referências)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--savedir", default=str(paths.SCRAPING_WORK / "spkrec-ecapa"),
                    help="cache do modelo ECAPA do SpeechBrain")
    args = ap.parse_args()

    D = Path(args.dir)
    refs = [Path(x) for x in args.refs] if args.refs else sorted(D.glob("ref_*.wav"))
    if not refs:
        sys.exit(f"[spk] nenhuma referência encontrada em {D} (use --refs ou crie ref_*.wav)")
    refset = {p.resolve() for p in refs}
    gens = ([Path(x) for x in args.gens] if args.gens
            else [p for p in sorted(D.glob("*.wav")) if p.resolve() not in refset])

    clf = _encoder(args.device, args.savedir)

    E = {p: _embed(clf, p, args.device) for p in refs}
    for i in range(len(refs)):
        for j in range(i + 1, len(refs)):
            c = float(torch.dot(E[refs[i]], E[refs[j]]))
            print(f"cos({refs[i].name}, {refs[j].name}) = {c:.3f}")

    print(f"\n# {len(gens)} geração(ões) vs {len(refs)} referência(s)  (dir={D})")
    for g in gens:
        if not g.exists():
            print(f"{g.name:38s} (ausente)")
            continue
        eg = _embed(clf, g, args.device)
        sims = "  ".join(f"{r.name}={float(torch.dot(eg, E[r])):.3f}" for r in refs)
        print(f"{g.name:38s} {sims}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
