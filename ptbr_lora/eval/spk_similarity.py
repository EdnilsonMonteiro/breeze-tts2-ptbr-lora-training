"""spk_similarity.py — similaridade de locutor (ECAPA, opcionalmente WavLM-SV).

Mede a identidade de voz: para cada áudio gerado, calcula o cosseno do embedding
contra cada referência. Regra prática: ~0,7+ = mesmo locutor; <0,3 = outra pessoa.

Recursos:
  * ECAPA (speechbrain/spkrec-ecapa-voxceleb) e, com --wavlm, também
    WavLM-SV (microsoft/wavlm-base-plus-sv) como segunda opinião.
  * --rms-match: iguala o RMS de referências e gerações antes de medir (evita o
    artefato "geração mais alta = cos menor"; no nosso dado r(cos, rms) = -0,40).
  * imprime a média por geração (SECS: similaridade contra TODAS as referências,
    que é o estimador estável; o cos contra uma única referência é ruidoso) e o
    teto intra-locutor (cos entre as próprias referências).

Uso:
  python spk_similarity.py --dir <pasta> [--refs a.wav b.wav] [--gens c.wav d.wav]
  python spk_similarity.py --dir teste_seeds/T0.9_k50_p1_cfg1 \
      --refs VozEdnilson.wav VozEdnilsonFalandoFrase.wav --wavlm --rms-match

Sem `--refs`/`--gens`, descobre automaticamente em `--dir`:
  referências = `ref_*.wav`; gerações = `*.wav` (exceto as referências).
Default de `--dir`: <PTBR_ARTIFACTS>/training/clone_out
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

_CORE = Path(__file__).resolve().parents[1] / "core"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))
import paths  # noqa: E402


def _load16k(path: Path, rms_target: float | None = None) -> np.ndarray:
    import librosa

    wav, _sr = librosa.load(str(path), sr=16000, mono=True)
    wav = np.asarray(wav, dtype=np.float32)
    if rms_target is not None:
        cur = float(np.sqrt(np.mean(wav ** 2)))
        if cur > 1e-6:
            wav = wav * (rms_target / cur)
    return wav


def _embed(clf, path: Path, device: str, rms_target: float | None = None) -> torch.Tensor:
    x = torch.from_numpy(_load16k(path, rms_target)).unsqueeze(0).to(device)
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


class _WavLM:
    """WavLM-SV (microsoft/wavlm-base-plus-sv) — segunda opinião de identidade."""

    def __init__(self, device: str):
        from transformers import AutoFeatureExtractor, WavLMForXVector

        self.fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        self.model = WavLMForXVector.from_pretrained(
            "microsoft/wavlm-base-plus-sv").to(device).eval()
        self.device = device

    def encode_batch(self, x: torch.Tensor) -> torch.Tensor:
        wav = x.squeeze(0).detach().cpu().numpy()
        inp = self.fe(wav, sampling_rate=16000, return_tensors="pt").to(self.device)
        with torch.no_grad():
            e = self.model(**inp).embeddings.squeeze(0)
        return (e / (e.norm() + 1e-9)).float()


def main() -> None:
    ap = argparse.ArgumentParser(description="Similaridade de locutor (ECAPA / WavLM-SV).")
    ap.add_argument("--dir", default=str(paths.TRAINING / "clone_out"),
                    help="pasta com referências e gerações")
    ap.add_argument("--refs", nargs="*", default=None,
                    help="arquivos de referência (default: ref_*.wav em --dir)")
    ap.add_argument("--gens", nargs="*", default=None,
                    help="áudios gerados (default: *.wav em --dir, exceto as referências)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--savedir", default=str(paths.SCRAPING_WORK / "spkrec-ecapa"),
                    help="cache do modelo ECAPA do SpeechBrain")
    ap.add_argument("--rms-match", action="store_true",
                    help="iguala o RMS de todas as amostras antes de medir (recomendado)")
    ap.add_argument("--wavlm", action="store_true",
                    help="mede também com WavLM-SV (microsoft/wavlm-base-plus-sv). ATENCAO: no "
                         "material deste projeto essa variante se mostrou POUCO DISCRIMINATIVA "
                         "(da 0,93 para um locutor DIFERENTE) -- use --others para conferir")
    ap.add_argument("--others", nargs="*", default=None,
                    help="audios de OUTROS locutores: imprime o piso de dissimilaridade de cada "
                         "metrica (controle obrigatorio antes de confiar no numero)")
    ap.add_argument("--exclude", nargs="*", default=None,
                    help="padroes (regex) de nomes a EXCLUIR da mediana. Descarta confusos "
                         "conhecidos: ex. 'regressao-en' (idioma) e clipes longos demais. "
                         "Vale a pena reportar a mediana COM e SEM eles.")
    args = ap.parse_args()

    D = Path(args.dir)
    refs = [Path(x) for x in args.refs] if args.refs else sorted(D.glob("ref_*.wav"))
    if not refs:
        sys.exit(f"[spk] nenhuma referência encontrada em {D} (use --refs ou crie ref_*.wav)")
    refset = {p.resolve() for p in refs}
    gens = ([Path(x) for x in args.gens] if args.gens
            else [p for p in sorted(D.glob("*.wav")) if p.resolve() not in refset])

    # RMS alvo: mediana do RMS das referências (igual para todos)
    rms_target = None
    if args.rms_match:
        vals = []
        for p in refs:
            y = _load16k(p)
            vals.append(float(np.sqrt(np.mean(y ** 2))))
        rms_target = float(np.median(vals))
        print(f"[spk] --rms-match: RMS alvo = {rms_target:.5f}")

    clf = _encoder(args.device, args.savedir)
    print("\n# ECAPA")

    E = {p: _embed(clf, p, args.device, rms_target) for p in refs}
    for i in range(len(refs)):
        for j in range(i + 1, len(refs)):
            c = float(torch.dot(E[refs[i]], E[refs[j]]))
            print(f"cos({refs[i].name}, {refs[j].name}) = {c:.3f}   <- teto intra-locutor")

    print(f"\n# {len(gens)} geração(ões) vs {len(refs)} referência(s)  (dir={D})")
    rows = []
    for g in gens:
        if not g.exists():
            print(f"{g.name:38s} (ausente)")
            continue
        eg = _embed(clf, g, args.device, rms_target)
        sims = [float(torch.dot(eg, E[r])) for r in refs]
        rows.append((g.name, sims))
        det = "  ".join(f"{r.name}={s:.3f}" for r, s in zip(refs, sims))
        print(f"{g.name:38s} SECS={np.mean(sims):.3f}  {det}")

    if args.wavlm and rows:
        print("\n# WavLM-SV (segunda opinião)")
        wl = _WavLM(args.device)
        Er = {p: wl.encode_batch(torch.from_numpy(_load16k(p, rms_target)).float())
              for p in refs}
        for i in range(len(refs)):
            for j in range(i + 1, len(refs)):
                print(f"cos({refs[i].name}, {refs[j].name}) = "
                      f"{float(torch.dot(Er[refs[i]], Er[refs[j]])):.3f}   <- teto intra-locutor")
        for name, _ in rows:
            g = next(p for p in gens if p.name == name)
            eg = wl.encode_batch(torch.from_numpy(_load16k(g, rms_target)).float())
            sims = [float(torch.dot(eg, Er[r])) for r in refs]
            print(f"{name:38s} SECS={np.mean(sims):.3f}  "
                  + "  ".join(f"{r.name}={s:.3f}" for r, s in zip(refs, sims)))

    if rows:
        allv = [np.mean(s) for _, s in rows]
        med = float(np.median(allv))
        sd = float(np.std(allv))
        print(f"\n[spk] mediana SECS = {med:.3f} | desvio = {sd:.3f} | n = {len(rows)}")
        if args.exclude:
            pats = [re.compile(p) for p in args.exclude]
            keep = [(n, float(np.mean(s))) for n, s in rows
                    if not any(p.search(n) for p in pats)]
            if keep:
                kv = [v for _, v in keep]
                print(f"[spk] mediana SECS (SEM {' '.join(args.exclude)}) = "
                      f"{float(np.median(kv)):.3f} | desvio = {float(np.std(kv)):.3f} "
                      f"| n = {len(keep)}")
                print("[spk] descartados por --exclude: "
                      + ", ".join(n for n, _ in rows
                                  if any(p.search(n) for p in pats)))
        print("[spk] use a MEDIANA e o DESVIO: a escolha por máximo é ruído (ver "
              "docs/CONSISTENCIA-DE-VOZ.md, secao 3.1)")

    if args.others:
        # Controle OBRIGATORIO: quanto a metrica da para um locutor DIFERENTE da referencia?
        # Se o piso de outro locutor ficar perto do valor da geracao, a metrica esta saturada
        # e o numero nao significa "mesma voz".
        print("\n# CONTROLE (outros locutores) — piso de dissimilaridade")
        for o in args.others:
            op = Path(o)
            if not op.is_file():
                print(f"{op.name:38s} (ausente)")
                continue
            eo = _embed(clf, op, args.device, rms_target)
            sims = [float(torch.dot(eo, E[r])) for r in refs]
            print(f"{op.name:38s} SECS={np.mean(sims):.3f}  "
                  + "  ".join(f"{r.name}={s:.3f}" for r, s in zip(refs, sims)))
        if args.wavlm:
            for o in args.others:
                op = Path(o)
                if not op.is_file():
                    continue
                eo = wl.encode_batch(torch.from_numpy(_load16k(op, rms_target)).float())
                sims = [float(torch.dot(eo, Er[r])) for r in refs]
                print(f"{op.name + ' (WavLM)':38s} SECS={np.mean(sims):.3f}  "
                      + "  ".join(f"{r.name}={s:.3f}" for r, s in zip(refs, sims)))
        print("[spk] regra: a metrica só é utilizavel se o piso (outro locutor) ficar BEM abaixo "
              "do valor da geracao; senao ela está saturada e o numero engana.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
