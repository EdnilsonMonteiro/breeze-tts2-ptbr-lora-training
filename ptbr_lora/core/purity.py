"""purity.py — pureza de locutor de um dataset de voz (analise pura: so numpy).

Pergunta que responde: "os clipes rotulados como o MESMO locutor sao de fato a mesma pessoa?"
Rotulos errados (diarizacao que junta duas pessoas, fala sobreposta, vinheta, convidado no
microfone do apresentador) ensinam o modelo de clonagem que a referencia NAO e confiavel.

Entrada: embeddings de locutor ja calculados (um por clipe, L2-normalizados) agrupados por
rotulo. Saida, por locutor:
  intra        media do cosseno entre pares de clipes do locutor (o "teto" da voz)
  loo_*        similaridade de cada clipe com o centroide dos DEMAIS clipes (leave-one-out)
  outliers     clipes muito abaixo da mediana do proprio locutor (provavel outra pessoa)
  split_*      teste de 2 grupos: o rotulo parece conter DUAS vozes distintas?
  nearest_*    locutor mais parecido no dataset (mesma pessoa com dois rotulos? vazamento?)
  status       OK / SUSPEITO / RUIM + motivos

Os limiares padrao foram calibrados para ECAPA (speechbrain/spkrec-ecapa-voxceleb) com audio
RMS-normalizado; outro modelo de embedding muda a escala -> ajuste via `PurityConfig`.
Este modulo nao le audio nem importa torch: e testavel isoladamente e reaproveitavel.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a")


# --------------------------------------------------------------------------- configuracao
@dataclass
class PurityConfig:
    min_clips: int = 4              # locutor com menos clipes: so listado, sem veredito
    # teto (intra) — ECAPA: estudio limpo ~0,65-0,80; abaixo de 0,45 raramente e uma voz so
    intra_warn: float = 0.55
    intra_bad: float = 0.45
    # outlier: loo < mediana - k * MAD robusto  OU  loo < piso absoluto
    outlier_k: float = 3.0
    outlier_floor: float = 0.30
    outlier_min_gap: float = 0.10   # e pelo menos 0,10 abaixo da mediana (evita falso alarme em voz muito coesa)
    outlier_frac_warn: float = 0.10
    outlier_frac_bad: float = 0.25
    # teste de 2 grupos (duas pessoas sob o mesmo rotulo)
    split_min_frac: float = 0.20    # o grupo menor precisa ter >= 20% dos clipes
    split_min_n: int = 3
    split_max_inter: float = 0.55   # centroides dos 2 grupos menos parecidos que isto
    split_margin: float = 0.12      # e cada grupo bem mais coeso por dentro do que entre si
    # duplicata (mesma pessoa com dois rotulos): cosseno entre centroides
    dup_threshold: float = 0.85

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- matematica
def l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-9)


def _mad(x: np.ndarray) -> float:
    med = float(np.median(x))
    return float(np.median(np.abs(x - med))) * 1.4826


def two_means(E: np.ndarray, iters: int = 20) -> np.ndarray:
    """K-means esferico com K=2, deterministico (inicia no par MENOS parecido). -> rotulos 0/1."""
    n = len(E)
    if n < 2:
        return np.zeros(n, dtype=np.int64)
    S = E @ E.T
    i, j = np.unravel_index(int(np.argmin(S)), S.shape)
    C = l2n(np.stack([E[i], E[j]]))
    lab = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        new = np.argmax(E @ C.T, axis=1)
        if _ > 0 and np.array_equal(new, lab):
            break
        lab = new
        for k in (0, 1):
            if np.any(lab == k):
                C[k] = l2n(E[lab == k].mean(axis=0))
    return lab


def _mean_offdiag(S: np.ndarray) -> float:
    n = len(S)
    if n < 2:
        return float("nan")
    return float((S.sum() - np.trace(S)) / (n * (n - 1)))


@dataclass
class SpeakerResult:
    speaker: str
    group: str = ""
    split: str = ""
    n: int = 0
    intra: float = float("nan")
    loo_median: float = float("nan")
    loo_p10: float = float("nan")
    n_outliers: int = 0
    outlier_frac: float = 0.0
    split_suspect: bool = False
    split_sizes: tuple = (0, 0)
    split_inter: float = float("nan")
    split_intra: tuple = (float("nan"), float("nan"))
    nearest: str = ""
    nearest_sim: float = float("nan")
    status: str = "OK"
    reasons: list = field(default_factory=list)
    clip_loo: list = field(default_factory=list)       # [(clip_id, loo_sim, is_outlier, cluster)]


def analyze_speaker(name: str, ids: list[str], E: np.ndarray, cfg: PurityConfig,
                    group: str = "", split: str = "") -> SpeakerResult:
    E = l2n(E)
    n = len(E)
    r = SpeakerResult(speaker=name, group=group, split=split, n=n)
    if n < 2:
        r.status = "POUCOS"
        r.reasons.append(f"so {n} clipe")
        return r
    S = E @ E.T
    r.intra = _mean_offdiag(S)
    tot = E.sum(axis=0)
    loo = np.array([float(np.dot(E[i], l2n(tot - E[i]))) for i in range(n)])
    r.loo_median = float(np.median(loo))
    r.loo_p10 = float(np.percentile(loo, 10))
    mad = max(_mad(loo), 0.02)
    gap = r.loo_median - loo
    out = ((gap > cfg.outlier_k * mad) & (gap >= cfg.outlier_min_gap)) | (loo < cfg.outlier_floor)
    r.n_outliers = int(out.sum())
    r.outlier_frac = r.n_outliers / n

    lab = two_means(E)
    sizes = (int((lab == 0).sum()), int((lab == 1).sum()))
    r.split_sizes = sizes
    if min(sizes) >= 1:
        c0, c1 = l2n(E[lab == 0].mean(axis=0)), l2n(E[lab == 1].mean(axis=0))
        r.split_inter = float(np.dot(c0, c1))
        r.split_intra = (_mean_offdiag(S[np.ix_(lab == 0, lab == 0)]),
                         _mean_offdiag(S[np.ix_(lab == 1, lab == 1)]))
        cross = float(S[np.ix_(lab == 0, lab == 1)].mean())
        small = min(sizes)
        r.split_suspect = bool(
            small >= cfg.split_min_n and small / n >= cfg.split_min_frac
            and r.split_inter < cfg.split_max_inter
            and min(r.split_intra) - cross >= cfg.split_margin)
    r.clip_loo = [(ids[i], float(loo[i]), bool(out[i]), int(lab[i])) for i in range(n)]

    if n < cfg.min_clips:
        r.status = "POUCOS"
        r.reasons.append(f"so {n} clipes (< {cfg.min_clips}); sem veredito")
        return r
    bad = warn = False
    if r.intra < cfg.intra_bad:
        bad = True
        r.reasons.append(f"teto {r.intra:.2f} < {cfg.intra_bad}")
    elif r.intra < cfg.intra_warn:
        warn = True
        r.reasons.append(f"teto {r.intra:.2f} < {cfg.intra_warn}")
    if r.outlier_frac >= cfg.outlier_frac_bad:
        bad = True
        r.reasons.append(f"{r.outlier_frac:.0%} clipes destoam")
    elif r.outlier_frac >= cfg.outlier_frac_warn:
        warn = True
        r.reasons.append(f"{r.outlier_frac:.0%} clipes destoam")
    if r.split_suspect:
        bad = True
        r.reasons.append(f"parece 2 vozes ({sizes[0]}+{sizes[1]} clipes, sim entre grupos {r.split_inter:.2f})")
    r.status = "RUIM" if bad else ("SUSPEITO" if warn else "OK")
    return r


def find_duplicates(results: list[SpeakerResult], centroids: dict[str, np.ndarray],
                    cfg: PurityConfig) -> list[dict]:
    """Pares de locutores com centroides muito parecidos (provavel mesma pessoa).
    Preenche nearest/nearest_sim em cada resultado; marca SUSPEITO se houver duplicata."""
    names = [r.speaker for r in results if r.speaker in centroids]
    if len(names) < 2:
        return []
    C = l2n(np.stack([centroids[n] for n in names]))
    S = C @ C.T
    np.fill_diagonal(S, -1.0)
    by = {r.speaker: r for r in results}
    pairs = []
    for a, name in enumerate(names):
        b = int(np.argmax(S[a]))
        by[name].nearest = names[b]
        by[name].nearest_sim = float(S[a, b])
    iu = np.argwhere(np.triu(S, 1) >= cfg.dup_threshold)
    for a, b in iu:
        ra, rb = by[names[a]], by[names[b]]
        leak = bool(ra.split and rb.split and ra.split != rb.split)
        pairs.append({"speaker_a": ra.speaker, "speaker_b": rb.speaker, "sim": round(float(S[a, b]), 4),
                      "group_a": ra.group, "group_b": rb.group, "split_a": ra.split, "split_b": rb.split,
                      "cross_split": leak})
        for r_, other in ((ra, rb), (rb, ra)):
            msg = f"igual a {other.speaker} ({S[a, b]:.2f})" + (" — splits diferentes: VAZAMENTO" if leak else "")
            r_.reasons.append(msg)
            if r_.status == "OK":
                r_.status = "SUSPEITO"
    pairs.sort(key=lambda p: -p["sim"])
    return pairs


def group_summary(results: list[SpeakerResult]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for g in sorted({r.group for r in results}):
        rs = [r for r in results if r.group == g and r.status != "POUCOS"]
        if not rs:
            continue
        intra = np.array([r.intra for r in rs])
        n_clips = sum(r.n for r in rs)
        out[g] = {
            "speakers": len(rs),
            "clips": n_clips,
            "intra_median": float(np.median(intra)),
            "intra_p10": float(np.percentile(intra, 10)),
            "ok": sum(r.status == "OK" for r in rs),
            "suspeito": sum(r.status == "SUSPEITO" for r in rs),
            "ruim": sum(r.status == "RUIM" for r in rs),
            "outlier_clips_frac": sum(r.n_outliers for r in rs) / max(1, n_clips),
        }
    return out


# --------------------------------------------------------------------------- entrada
_FIELD = re.compile(r"{([^{}]+)}")


def render(template: str, row: dict) -> str:
    """'{tag}:{speaker}' com os campos da linha (erro claro se faltar campo)."""
    def sub(m):
        k = m.group(1)
        if k not in row:
            raise KeyError(f"campo '{k}' nao existe (campos: {', '.join(sorted(row))})")
        return str(row[k])
    return _FIELD.sub(sub, template)


@dataclass
class Clip:
    id: str
    audio: Path
    speaker: str
    group: str = ""
    split: str = ""


def read_rows(path: Path) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else data.get("items", [])
    with open(path, encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        delim = "|" if sample.count("|") > sample.count(",") else ("\t" if "\t" in sample.splitlines()[0] else ",")
        return list(csv.DictReader(f, delimiter=delim))


def clips_from_manifest(path: Path, audio: str, speaker: str, group: str = "", split: str = "",
                        clip_id: str = "", root: Path | None = None) -> list[Clip]:
    rows = read_rows(path)
    root = Path(root) if root else Path(path).resolve().parent
    out = []
    for k, row in enumerate(rows):
        a = Path(render(audio, row))
        out.append(Clip(id=render(clip_id, row) if clip_id else a.stem or str(k),
                        audio=a if a.is_absolute() else root / a,
                        speaker=render(speaker, row),
                        group=render(group, row) if group else "",
                        split=render(split, row) if split else ""))
    return out


def clips_from_folders(root: Path, group: str = "") -> list[Clip]:
    """Layout simples para quem traz o proprio audio: <root>/<locutor>/**/*.wav"""
    root = Path(root)
    out = []
    for spk_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(spk_dir.rglob("*")):
            if f.suffix.lower() in AUDIO_EXTS:
                out.append(Clip(id=f"{spk_dir.name}/{f.relative_to(spk_dir).as_posix()}", audio=f,
                                speaker=spk_dir.name, group=group or root.name))
    return out


def clips_from_training_dir(tdir: Path) -> list[Clip]:
    """Preset do projeto: <training_dir>/manifest.csv + speaker_table.csv (split) + wavs24/<idx>.wav"""
    tdir = Path(tdir)
    split_of = {}
    st = tdir / "speaker_table.csv"
    if st.is_file():
        for row in read_rows(st):
            split_of[row["speaker"]] = row.get("split", "")
    clips = clips_from_manifest(tdir / "manifest.csv", audio="wavs24/{idx}.wav", speaker="{speaker}",
                                group="{corpus}", clip_id="{idx}", root=tdir)
    for c in clips:
        c.split = split_of.get(c.speaker, "")
    return clips


def sample_per_speaker(clips: list[Clip], max_per_speaker: int, seed: int = 0) -> list[Clip]:
    """Ate N clipes por locutor, sorteio deterministico (o mesmo a cada execucao)."""
    if max_per_speaker <= 0:
        return list(clips)
    by: dict[str, list[Clip]] = {}
    for c in clips:
        by.setdefault(c.speaker, []).append(c)
    rng = np.random.default_rng(seed)
    out = []
    for spk in sorted(by):
        lst = sorted(by[spk], key=lambda c: c.id)
        if len(lst) > max_per_speaker:
            idx = sorted(rng.choice(len(lst), size=max_per_speaker, replace=False).tolist())
            lst = [lst[i] for i in idx]
        out.extend(lst)
    return out


# --------------------------------------------------------------------------- execucao completa
def analyze(clips: list[Clip], emb: dict[str, np.ndarray], cfg: PurityConfig):
    """-> (results, duplicates, summary). `emb` = {clip.id: vetor}; clipes sem vetor sao ignorados."""
    by: dict[str, list[Clip]] = {}
    for c in clips:
        if c.id in emb:
            by.setdefault(c.speaker, []).append(c)
    results, centroids = [], {}
    for spk in sorted(by):
        cs = by[spk]
        E = l2n(np.stack([emb[c.id] for c in cs]))
        r = analyze_speaker(spk, [c.id for c in cs], E, cfg, group=cs[0].group, split=cs[0].split)
        results.append(r)
        if r.n >= cfg.min_clips:
            keep = [i for i, (_, _, o, _) in enumerate(r.clip_loo) if not o] or list(range(len(cs)))
            centroids[spk] = l2n(E[keep].mean(axis=0))
    dups = find_duplicates(results, centroids, cfg)
    return results, dups, group_summary(results)
