"""speaker_identity.py — identidade de locutor em audio diarizado (podcast, entrevista, video).

Modulo puro (so numpy). A diarizacao da um rotulo POR ARQUIVO ("SPEAKER_00" do episodio A nao tem
relacao com "SPEAKER_00" do episodio B) e erra de tres jeitos que estragam um dataset de clonagem:
  1. clipe com DUAS pessoas (troca de locutor no meio, ou a margem do corte pega o vizinho);
  2. rotulo que junta duas pessoas;
  3. a MESMA pessoa em episodios diferentes com rotulos diferentes (o apresentador do programa)
     -> vira "duas vozes" no treino e pode cair em treino E validacao (vazamento).
Este modulo decide, a partir de embeddings de locutor ja calculados:
  * `edge_bleed`          clipe cuja margem invade fala exclusiva de outro locutor (via RTTM);
  * `refine_label`        limpa um rotulo: tira clipes intrusos, separa rotulo com 2 vozes;
  * `complete_linkage`    agrupa rotulos de episodios diferentes que sao a mesma pessoa
                          (ligacao COMPLETA: todos os pares do grupo acima do limiar -> conservador);
  * `calibrate_link_threshold` limiar a partir de pares sabidamente DIFERENTES (outro corpus);
  * `episode_groups`      episodios que compartilham alguem viram um grupo so (unidade de split).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from purity import PurityConfig, analyze_speaker, l2n


# --------------------------------------------------------------------------- clipe x RTTM
def edge_bleed(spans: list[list[float]], pad: float, own: str,
               rttm: list[tuple[float, float, str]], min_overlap: float = 0.02) -> bool:
    """True se a janela [inicio-pad, fim+pad] do clipe toca fala EXCLUSIVA de outro locutor."""
    a, b = spans[0][0], spans[-1][1]
    a2, b2 = max(0.0, a - pad), b + pad
    for s, e, spk in rttm:
        if spk == own or e <= a2 or s >= b2:
            continue
        if min(e, b2) - max(s, a2) >= min_overlap:
            return True
    return False


# --------------------------------------------------------------------------- limpeza de um rotulo
@dataclass
class Refined:
    name: str
    kept: dict = field(default_factory=dict)       # sub-rotulo -> [clip_id]
    dropped: dict = field(default_factory=dict)    # clip_id -> motivo
    note: str = ""


def refine_label(name: str, ids: list[str], E: np.ndarray, cfg: PurityConfig,
                 min_sub_clips: int = 8, min_sub_intra: float = 0.55) -> Refined:
    """Tira intrusos; se o rotulo tem 2 vozes, vira `name#a` e `name#b` (cada uma so se for coesa)."""
    out = Refined(name=name)
    if len(ids) < 2:
        out.kept[name] = list(ids)
        return out
    r = analyze_speaker(name, ids, E, cfg)
    if not r.split_suspect:
        for cid, _sim, is_out, _k in r.clip_loo:
            if is_out:
                out.dropped[cid] = "destoa do locutor"
        out.kept[name] = [c for c, _s, o, _k in r.clip_loo if not o]
        return out
    # 2 vozes: cada grupo e limpo CONTRA SI MESMO (contra o rotulo todo, o grupo menor pareceria intruso)
    lab = {cid: k for cid, _s, _o, k in r.clip_loo}
    pos = {cid: i for i, cid in enumerate(ids)}
    notes = []
    for k, suf in ((0, "#a"), (1, "#b")):
        sub = [c for c in ids if lab[c] == k]
        good = sub
        if len(sub) >= 2:
            rs = analyze_speaker(name + suf, sub, E[[pos[c] for c in sub]], cfg)
            for cid, _s, o, _k in rs.clip_loo:
                if o:
                    out.dropped[cid] = "destoa do locutor"
            good = [c for c, _s, o, _k in rs.clip_loo if not o]
            if len(good) >= max(2, min_sub_clips) and rs.intra >= min_sub_intra:
                out.kept[name + suf] = good
                notes.append(f"{suf}={len(good)}")
                continue
        for c in good:
            out.dropped[c] = "rotulo com 2 vozes (grupo pequeno/incoerente)"
    out.note = "2 vozes -> " + (", ".join(notes) or "nenhum grupo aproveitavel")
    return out


# --------------------------------------------------------------------------- ligacao entre episodios
def complete_linkage(C: np.ndarray, thr: float) -> np.ndarray:
    """Agrupamento aglomerativo de ligacao COMPLETA sobre cosseno: dois grupos so se unem se TODOS
    os pares entre eles tiverem similaridade >= thr. Deterministico. -> rotulo por linha."""
    C = l2n(C)
    n = len(C)
    S = C @ C.T
    clusters = [[i] for i in range(n)]
    # similaridade de ligacao completa entre clusters = minimo dos pares
    L = S.copy()
    np.fill_diagonal(L, -np.inf)
    alive = list(range(n))
    while len(alive) > 1:
        sub = L[np.ix_(alive, alive)]
        k = int(np.argmax(sub))
        i, j = divmod(k, len(alive))
        if sub[i, j] < thr:
            break
        a, b = alive[i], alive[j]
        if a > b:
            a, b = b, a
        clusters[a] += clusters[b]
        clusters[b] = []
        L[a, :] = np.minimum(L[a, :], L[b, :])
        L[:, a] = L[a, :]
        L[a, a] = -np.inf
        alive.remove(b)
    lab = np.zeros(n, dtype=np.int64)
    for k, a in enumerate(sorted(alive, key=lambda x: min(clusters[x]))):
        for i in clusters[a]:
            lab[i] = k
    return lab


def calibrate_link_threshold(diff_sims: np.ndarray, floor: float = 0.75, margin: float = 0.03,
                             q: float = 99.9) -> float:
    """Limiar = max(piso, percentil q dos pares de pessoas DIFERENTES + margem)."""
    diff_sims = np.asarray(diff_sims, dtype=np.float64)
    if diff_sims.size == 0:
        return floor
    return float(max(floor, np.percentile(diff_sims, q) + margin))


def centroid_pair_sims(groups: list[np.ndarray]) -> np.ndarray:
    """Similaridades entre centroides de grupos (triangulo superior)."""
    C = l2n(np.stack([l2n(g).mean(axis=0) for g in groups]))
    S = C @ C.T
    return S[np.triu_indices(len(C), 1)]


def split_half_sims(groups: list[np.ndarray], seed: int = 0) -> np.ndarray:
    """Mesma pessoa: centroide de metade dos clipes x centroide da outra metade (referencia 'igual')."""
    rng = np.random.default_rng(seed)
    out = []
    for g in groups:
        if len(g) < 6:
            continue
        p = rng.permutation(len(g))
        h = len(g) // 2
        a, b = l2n(l2n(g[p[:h]]).mean(axis=0)), l2n(l2n(g[p[h:]]).mean(axis=0))
        out.append(float(a @ b))
    return np.array(out)


# --------------------------------------------------------------------------- grupos de split
def episode_groups(members: dict[str, list[str]]) -> dict[str, str]:
    """members: id_global -> [episodio,...]. Episodios ligados por alguem em comum = um grupo.
    -> episodio -> 'G###' (ordenado pelo 1o episodio, deterministico)."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for eps in members.values():
        eps = sorted(set(eps))
        for e in eps:
            find(e)
        for e in eps[1:]:
            ra, rb = find(eps[0]), find(e)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    roots = sorted({find(e) for e in parent})
    gid = {r: f"G{k + 1:03d}" for k, r in enumerate(roots)}
    return {e: gid[find(e)] for e in parent}
