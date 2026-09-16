"""Fase 3 - diarizacao (pipeline minimo oficial da pyannote):
pyannote/segmentation-3.0 (gated, aceito no HF) + SpeechBrain ECAPA (nao-gated)
+ clustering aglomerativo em 2 passes (longos agrupam; curtos herdam por centroide;
clusters < MIN_CLUSTER_S sao dissolvidos).
Saida: work/diar/{tag}.json (schema consumido por 02_slice.py) + work/rttm/{tag}.rttm

Por que nao pyannote/speaker-diarization-3.1 de pipeline pronto: no pyannote.audio 4.x
o from_pretrained resolve para o repo gated 'speaker-diarization-community-1' (mais um
consentimento no HF) e o 3.3.2 (stack classica) e incompativel com torchaudio>=2.9.
Este script reproduz a receita minima publicada pela propria equipe pyannote.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
import paths  # noqa: E402

ROOT = paths.ARTIFACTS
WORK = ROOT / "dataScrapping" / "work"
RTTM = WORK / "rttm"
DIAR = WORK / "diar"
RTTM.mkdir(parents=True, exist_ok=True)
DIAR.mkdir(parents=True, exist_ok=True)

SR = 16000
FRAME_ACT = 0.6      # limiar de ativacao do segmentation
MIN_SPAN_S = 0.40    # span exclusivo minimo
EMBED_MIN_S = 2.0   # span minimo p/ embedding confiavel (passo 1)
CLUSTER_DIST = 0.45  # distancia 1-cos no linkage average
KEEP_ASSIGN_SIM = 0.45  # sim. cos. minima p/ span curto herdar cluster
MIN_CLUSTER_S = 60.0    # cluster com menos fala que isso eh dissolvido
NEAR_ASSIGN_S = 5.0  # spans curtos herdam cluster do vizinho temporais

hf_token = None
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    if line.startswith("HF_TOKEN"):
        hf_token = line.split("=", 1)[1].strip()

from pyannote.audio import Model
from speechbrain.inference.speaker import EncoderClassifier

t0 = time.time()
seg_model = Model.from_pretrained("pyannote/segmentation-3.0", token=hf_token)
seg_model.to(torch.device("cuda")).eval()
print(f"[diar2] segmentation-3.0 carregado ({time.time() - t0:.0f}s)", flush=True)

embedder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=str(WORK / "spkrec-ecapa"),
    run_opts={"device": "cuda"})
print("[diar2] ECAPA carregado", flush=True)

WIN = int(round(seg_model.specifications.duration))  # 10.0s


@torch.inference_mode()
def segment_probs(wav: np.ndarray) -> np.ndarray:
    step = SR * WIN
    n_win = len(wav) // step
    out = []
    B = 32
    for i in range(0, n_win, B):
        batch = torch.from_numpy(
            wav[i * step:min(i + B, n_win) * step]
            .reshape(-1, step)).unsqueeze(1).cuda()
        p = seg_model(batch).exp().float().cpu().numpy()  # saida = log-probs -> probs
        out.append(p)
    probs = np.concatenate(out, axis=0)  # (n_win, F, S)
    n_frames = probs.shape[1]
    frame_step = WIN / n_frames
    probs = probs.reshape(n_win * n_frames, -1)
    return probs, frame_step


@torch.inference_mode()
def embed(span_wav: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(span_wav).float().unsqueeze(0).cuda()
    e = embedder.encode_batch(t).squeeze().cpu().numpy()
    e = e / (np.linalg.norm(e) + 1e-9)
    return e


episodes = json.loads((WORK / "episodes.json").read_text(encoding="utf-8"))
for ep in episodes:
    tag = ep["tag"]
    t1 = time.time()
    wav, sr = sf.read(ep["w16k"], dtype="float32")
    assert sr == SR
    probs, frame_step = segment_probs(wav)
    active = probs > FRAME_ACT
    n_active = active.sum(axis=1)

    # spans exclusivos (sem overlap) por alto-falante bruto
    spans, raw_hours = [], {}
    for s in range(active.shape[1]):
        raw_hours[s] = float(active[:, s].sum() * frame_step) / 3600
        mask = active[:, s] & (n_active == 1)
        idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
        for a, b in idx.reshape(-1, 2):
            if (b - a) * frame_step >= MIN_SPAN_S:
                spans.append([a * frame_step, b * frame_step, s])
    spans.sort()
    print(f"[diar2] {tag}: {len(spans)} spans exclusivos de {len(raw_hours)} vozes brutas", flush=True)

    # embeddings por span (clusterizacao de identidade)
    embs, emb_spans = [], []
    for a, b, s in spans:
        if b - a >= EMBED_MIN_S:
            embs.append(embed(wav[int(a * SR):int(b * SR)]))
            emb_spans.append([a, b, s])
    print(f"[diar2] {tag}: {len(embs)} embeddings em {time.time() - t1:.0f}s", flush=True)

    # --- clustering em 2 passes ---
    # passo 1: so spans longos (embedding confiavel) -> linkage average
    D = 1.0 - np.clip(np.array(embs) @ np.array(embs).T, -1, 1)
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    labels = fcluster(Z, t=CLUSTER_DIST, criterion="distance")

    cents, spans_by_cluster = {}, {}
    for i, c in enumerate(set(labels)):
        vecs = [embs[j] for j in range(len(embs)) if labels[j] == c]
        ivs = [emb_spans[j] for j in range(len(embs)) if labels[j] == c]
        m = np.mean(vecs, axis=0)
        cents[c] = m / (np.linalg.norm(m) + 1e-9)
        spans_by_cluster[c] = [[a, b] for a, b, _ in ivs]

    # dissolve clusters pequenos: reatribui pelo centroide se proximo, senao descarta
    while True:
        small = [c for c, ivs in spans_by_cluster.items()
                 if sum(b - a for a, b in ivs) < MIN_CLUSTER_S]
        if not small or len(spans_by_cluster) <= 1:
            break
        c = min(small, key=lambda k: sum(b - a for a, b in spans_by_cluster[k]))
        ivs = spans_by_cluster.pop(c)
        cents.pop(c)
        others = list(spans_by_cluster)
        if not others:
            spans_by_cluster[c] = ivs
            break
        M = np.array([cents[o] for o in others])
        for j, (aa, bb, _s) in enumerate(emb_spans):
            if any(abs(aa - a) < 1e-6 and abs(bb - b) < 1e-6 for a, b in ivs):
                sims = M @ embs[j]
                k = int(np.argmax(sims))
                if sims[k] >= KEEP_ASSIGN_SIM:
                    spans_by_cluster[others[k]].append([aa, bb])
        for o in others:
            mem = [embs[j] for j in range(len(embs))
                   if labels[j] == o and any(abs(emb_spans[j][0] - a) < 1e-6
                                             and abs(emb_spans[j][1] - b) < 1e-6
                                             for a, b in spans_by_cluster[o])]
            if mem:
                mm = np.mean(mem, axis=0)
                cents[o] = mm / (np.linalg.norm(mm) + 1e-9)

    # passo 2: spans curtos herdam cluster por centroide
    others = list(spans_by_cluster)
    M = np.array([cents[o] for o in others]) if others else np.zeros((0, len(embs[0]) if embs else 0))
    names = {c: f"SPEAKER_{k:02d}" for k, c in enumerate(sorted(
        spans_by_cluster, key=lambda c: -sum(b - a for a, b in spans_by_cluster[c])))}
    final = []
    for c, ivs in spans_by_cluster.items():
        for a, b in ivs:
            final.append((a, b, names[c]))
    for a, b, _ in spans:
        if b - a >= EMBED_MIN_S:
            continue
        if not len(M):
            continue
        e = embed(wav[int(a * SR):int(b * SR)])
        sims = M @ e
        k = int(np.argmax(sims))
        if sims[k] >= KEEP_ASSIGN_SIM:
            final.append((a, b, names[others[k]]))

    clean = {}
    for a, b, spk in final:
        clean.setdefault(spk, []).append([a, b])
    rep = {}
    for spk, ivs in clean.items():
        h = round(sum(b - a for a, b in ivs) / 3600, 2)
        rep[spk] = {"h_raw": h, "h_clean": h}  # fallback: sem overlap por definicao (spans exclusivos)
    (DIAR / f"{tag}.json").write_text(json.dumps(
        {"tag": tag, "elapsed_s": round(time.time() - t1), "speakers": rep,
         "clean": {k: [[round(a, 3), round(b, 3)] for a, b in v] for k, v in clean.items()}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    with (RTTM / f"{tag}.rttm").open("w", encoding="utf-8") as fh:
        for a, b, spk in sorted(final):
            fh.write(f"SPEAKER {tag} 1 {a:.3f} {b - a:.3f} <NA> <NA> {spk} <NA> <NA>\n")
    print(f"[diar2] {tag}: {len(clean)} vozes | "
          + " | ".join(f"{k}: {v['h_clean']:.2f}h" for k, v in rep.items())
          + f" | {time.time() - t1:.0f}s", flush=True)

print("[diar2] OK", flush=True)
