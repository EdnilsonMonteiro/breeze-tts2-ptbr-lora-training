"""tagarela_speakers.py — deriva locutores do TAGARELA por clustering de embeddings
(ECAPA) DENTRO de cada show, e um ref_map (idx -> ref_idx) com guarda de similaridade.

Os clipes do TAGARELA ja sao mono-locutor (diarizados na fonte); aqui so agrupamos
por voz via ECAPA (mesma receita de clustering do dataScrapping/01_diarize.py).

Saidas:
  datasets/tagarela/embeddings.npz   (cache resumivel)
  datasets/tagarela/speakers.jsonl   (idx -> 'tag:<show>:<NN>')
  datasets/tagarela/ref_map.jsonl    (idx -> ref_idx do MESMO locutor, cos>=limiar)

Uso: python scripts/tagarela_speakers.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
import paths  # noqa: E402

TAG = paths.DATASETS_ROOT / "tagarela"
WAVS = TAG / "wavs"
SEL = TAG / "selection.jsonl"
EMB_P = TAG / "embeddings.npz"
SPK_P = TAG / "speakers.jsonl"
REF_P = TAG / "ref_map.jsonl"
ECAPA_DIR = paths.SCRAPING_WORK / "spkrec-ecapa"

CLUSTER_DIST = 0.45   # distancia 1-cos no linkage average (cos>=0.55 no cluster)
REF_MIN_COS = 0.55    # similaridade minima alvo<->ref
MIN_CLUSTER = 2       # clusters com menos clipes sao dissolvidos


def show_of(key: str) -> str:
    for p in key.split("/"):
        if p.startswith("show_"):
            return p
    return "unknown"


def main() -> None:
    recs = [json.loads(l) for l in SEL.read_text(encoding="utf-8").splitlines() if l.strip()]
    idxs = [r["idx"] for r in recs]
    show = {r["idx"]: show_of(r["key"]) for r in recs}
    by_show: dict[str, list[str]] = {}
    for i in idxs:
        by_show.setdefault(show[i], []).append(i)
    print(f"[spk] {len(idxs)} clipes | {len(by_show)} shows", flush=True)

    from speechbrain.inference.speaker import EncoderClassifier

    clf = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(ECAPA_DIR), run_opts={"device": "cuda"})

    E: dict[str, np.ndarray] = {}
    if EMB_P.exists():
        z = np.load(EMB_P)
        E = {k: z[k] for k in z.files}
        print(f"[spk] embeddings em cache: {len(E)}", flush=True)
    t0 = time.time()
    n = 0
    for i in idxs:
        if i in E:
            continue
        w, sr = sf.read(str(WAVS / f"{i}.wav"), dtype="float32", always_2d=True)
        x = torch.from_numpy(w.mean(1)).unsqueeze(0).cuda()
        with torch.no_grad():
            e = clf.encode_batch(x).squeeze().cpu().numpy()
        E[i] = e / (np.linalg.norm(e) + 1e-9)
        n += 1
        if n % 5000 == 0:
            np.savez(EMB_P, **E)
            print(f"[spk] {n} embeddings ({time.time()-t0:.0f}s)", flush=True)
    np.savez(EMB_P, **E)
    print(f"[spk] embeddings prontos: {len(E)} em {time.time()-t0:.0f}s", flush=True)

    spk_rows, ref_rows = [], []
    n_spk = 0
    intra = []
    for sh, lst in by_show.items():
        if len(lst) < 2:
            for i in lst:
                spk_rows.append({"idx": i, "speaker": f"tag:{sh}:00"})
                ref_rows.append({"idx": i, "ref_idx": i, "cos": 1.0})
            n_spk += 1
            continue
        M = np.array([E[i] for i in lst])
        D = 1.0 - np.clip(M @ M.T, -1, 1)
        np.fill_diagonal(D, 0.0)
        Z = linkage(squareform(D, checks=False), method="average")
        labels = fcluster(Z, t=CLUSTER_DIST, criterion="distance")
        for c in sorted(set(labels)):
            members = [lst[j] for j in range(len(lst)) if labels[j] == c]
            if len(members) < MIN_CLUSTER:
                for i in members:
                    spk_rows.append({"idx": i, "speaker": f"tag:{sh}:99"})
                    ref_rows.append({"idx": i, "ref_idx": i, "cos": 1.0})
                n_spk += 1
                continue
            name = f"tag:{sh}:{n_spk:03d}"
            n_spk += 1
            Mm = np.array([E[i] for i in members])
            S = Mm @ Mm.T
            iu = np.triu_indices(len(members), 1)
            if len(members) > 1:
                intra.append(float(S[iu].mean()))
            for j, i in enumerate(members):
                spk_rows.append({"idx": i, "speaker": name})
                sims = S[j].copy()
                sims[j] = -1
                order = np.argsort(-sims)
                ref_i, ref_cos = i, 1.0
                for k in order:
                    if sims[k] >= REF_MIN_COS and members[k] != i:
                        ref_i, ref_cos = members[k], float(sims[k])
                        break
                ref_rows.append({"idx": i, "ref_idx": ref_i, "cos": round(ref_cos, 3)})

    SPK_P.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in spk_rows),
                     encoding="utf-8")
    REF_P.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in ref_rows),
                     encoding="utf-8")
    with_ref = sum(1 for r in ref_rows if r["ref_idx"] != r["idx"])
    print(f"[spk] FIM: {n_spk} locutores | {with_ref}/{len(ref_rows)} com ref de outro clipe "
          f"({100*with_ref/len(ref_rows):.1f}%) | cos intra-cluster medio "
          f"{np.mean(intra):.3f}" if intra else "[spk] FIM")
    print(f"[spk] -> {SPK_P.name}, {REF_P.name}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
