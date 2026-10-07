"""splits.py — split treino/val/teste SEM vazamento, por GRUPO. Modulo puro (sem torch).

Problemas do split antigo (auditoria 2026-09):
  * o "locutor" do tagarela/podcast e `programa:N` (diarizacao por episodio) -> o MESMO
    programa/voz aparecia em treino e em val/test com ids de locutor diferentes;
  * guloso por locutor embaralhado ate 90/95 % -> val/test com tamanhos ruins e sem
    garantia de conter todos os corpora.

Aqui o GRUPO e a unidade de split:
  tagarela/podcast -> `speaker.rsplit(":", 1)[0]` (programa/episodio inteiro)
  demais           -> speaker
Atribuicao por deficit (grupos grandes primeiro), estratificada por corpus, seed fixa.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict

SPLITS = ("train", "val", "test")
DEFAULT_FRACS = (0.90, 0.05, 0.05)
GROUPED_CORPORA = {"tagarela", "podcast"}


def group_key(rec: dict) -> str:
    corpus = rec.get("corpus", "tata")
    if rec.get("group"):                                # grupo explicito (ex.: episodios que dividem um locutor)
        return f"{corpus}/{rec['group']}"
    spk = rec.get("speaker") or rec["idx"]
    if corpus in GROUPED_CORPORA and ":" in spk:
        spk = spk.rsplit(":", 1)[0]
    return f"{corpus}/{spk}"


def _tie(seed: int, key: str) -> str:
    return hashlib.sha1(f"{seed}|{key}".encode()).hexdigest()


def make_splits(recs: list[dict], fracs: tuple[float, float, float] = DEFAULT_FRACS,
                seed: int = 42, tolerance: float = 0.5) -> dict[str, list[str]]:
    """Retorna {split: [idx,...]}; deterministico; nenhum grupo em >1 split."""
    assert abs(sum(fracs) - 1.0) < 1e-6, fracs
    out: dict[str, list[str]] = {s: [] for s in SPLITS}
    by_corpus: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for r in recs:
        by_corpus[r.get("corpus", "tata")][group_key(r)].append(r["idx"])

    for corpus in sorted(by_corpus):
        groups = by_corpus[corpus]
        total = sum(len(v) for v in groups.values())
        target = {s: f * total for s, f in zip(SPLITS, fracs)}
        cur = {s: 0 for s in SPLITS}
        assign: dict[str, str] = {}
        order = sorted(groups, key=lambda g: (-len(groups[g]), _tie(seed, g)))
        for g in order:
            size = len(groups[g])
            cands = [s for s in SPLITS if cur[s] + size <= target[s] * (1 + tolerance)]
            if not cands:
                cands = ["train"]
            dest = max(cands, key=lambda s: (target[s] - cur[s], s == "train"))
            assign[g] = dest
            cur[dest] += size
        # garante val e test nao vazios quando ha grupos suficientes (>= 3)
        if len(groups) >= 3:
            for need in ("val", "test"):
                if cur[need] == 0:
                    donors = sorted((g for g in groups if assign[g] == "train"),
                                    key=lambda g: (len(groups[g]), _tie(seed, g)))
                    if donors and len([g for g in groups if assign[g] == "train"]) > 1:
                        g = donors[0]
                        cur["train"] -= len(groups[g])
                        assign[g] = need
                        cur[need] += len(groups[g])
        for g, dest in assign.items():
            out[dest].extend(groups[g])
    for s in SPLITS:
        out[s] = sorted(out[s])
    return out


def audit_leaks(recs: list[dict], splits: dict[str, list[str]]) -> dict:
    """Grupos/locutores presentes em mais de um split (ambos devem ser vazios)."""
    by_idx = {r["idx"]: r for r in recs}
    g_split: dict[str, set[str]] = defaultdict(set)
    s_split: dict[str, set[str]] = defaultdict(set)
    for name, idxs in splits.items():
        for i in idxs:
            r = by_idx.get(i)
            if r is None:
                continue
            g_split[group_key(r)].add(name)
            s_split[f"{r.get('corpus', 'tata')}/{r.get('speaker') or r['idx']}"].add(name)
    return {
        "group_leaks": sorted(g for g, v in g_split.items() if len(v) > 1),
        "speaker_leaks": sorted(s for s, v in s_split.items() if len(v) > 1),
    }


def split_stats(recs: list[dict], splits: dict[str, list[str]]) -> dict:
    by_idx = {r["idx"]: r for r in recs}
    stats: dict = {}
    for name, idxs in splits.items():
        per: dict[str, dict] = {}
        for i in idxs:
            r = by_idx.get(i)
            if r is None:
                continue
            d = per.setdefault(r.get("corpus", "tata"), {"n": 0, "s": 0.0, "groups": set()})
            d["n"] += 1
            d["s"] += float(r.get("dur_proc_s", 0.0))
            d["groups"].add(group_key(r))
        stats[name] = {c: {"n": d["n"], "hours": round(d["s"] / 3600, 3), "groups": len(d["groups"])}
                       for c, d in sorted(per.items())}
        stats[name]["_total"] = {"n": len(idxs)}
    return stats


def write_splits(splits: dict[str, list[str]], out_dir) -> None:
    from pathlib import Path

    d = Path(out_dir)
    for name, lst in splits.items():
        (d / f"splits_{name}.txt").write_text("\n".join(lst), encoding="utf-8")


def load_split_file(out_dir, name: str) -> list[str]:
    from pathlib import Path

    p = Path(out_dir) / f"splits_{name}.txt"
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
