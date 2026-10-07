"""voices.py — base DIVERSA em locutores: selecao, splits por voz e pesos de amostragem por MASSA.

Modulo puro (sem torch). Motivacao (diagnostico r73_01, 2026-10-01): com 6 locutores CML (~24 %
das amostras) e 32 CETUC (~25 %), o adapter DECOROU essas vozes — o SECS dos locutores do treino
subia (ate acima do teto real x real) enquanto o de locutores nao vistos caia a cada passo. Aqui:

  1. `select_voices`     : so entra "voz" com >= min_clips clipes utilizaveis; cada voz e cortada em
                          `cap` clipes (determinístico) -> muitas vozes, poucas horas por voz;
  2. `make_voice_splits`: val/test = GRUPOS inteiros (show/locutor) nunca vistos, em numero fixo por
                          corpus, com teto de clipes por voz (avaliacao equilibrada);
  3. `speaker_mass_weights`: probabilidade de amostragem POR VOZ com teto de massa (default 1 %)
                          e teto por grupo (show, 2 %), com participacao fixa por corpus.
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass, field

import splits as SPL

UNASSIGNED_SUFFIX = ":99"

DEFAULT_CAPS = {"cetuc": 120, "cml_pt": 120, "podcast": 40, "tagarela": 40, "tata": 120,
                "mls_pt": 60, "cv_pt": 25}          # mls_pt / cv_pt: corpora extras (ingest_extra_corpus.py)
DEFAULT_CAP = 40
# vozes mantidas fora do treino (val, test) — grupos inteiros, por corpus
DEFAULT_HOLDOUT_GROUPS = {"cetuc": (3, 3), "cml_pt": (3, 3), "podcast": (2, 2),
                          "tagarela": (16, 16), "tata": (0, 0), "mls_pt": (6, 6), "cv_pt": (8, 8)}
# fracao da MASSA de amostragem por corpus (renormalizada entre os corpora presentes)
DEFAULT_CORPUS_SHARE = {"tagarela": 0.50, "podcast": 0.15, "cml_pt": 0.15, "cetuc": 0.15,
                        "tata": 0.02, "mls_pt": 0.10, "cv_pt": 0.15}   # renormalizado entre os PRESENTES


@dataclass
class VoiceCfg:
    min_clips: int = 12                       # clipes utilizaveis minimos por voz
    min_dur_s: float = 2.5                    # = refs.MIN_REF_S (todo clipe pode virar referencia)
    max_dur_s: float = 10.2                   # = refs.MAX_REF_S
    min_text_chars: int = 8
    caps: dict = field(default_factory=lambda: dict(DEFAULT_CAPS))
    default_cap: int = DEFAULT_CAP
    holdout_groups: dict = field(default_factory=lambda: dict(DEFAULT_HOLDOUT_GROUPS))
    min_holdout_clips: int = 15               # grupo precisa ter isso (apos o corte) p/ ser val/test
    heldout_cap: int = 24                     # clipes por VOZ em val/test
    seed: int = 42
    exclude_corpora: tuple = ()
    balance_gender: tuple = ("cv_pt",)        # corpora com vozes M/F em numero IGUAL (campo `gender`)


def _h(seed: int, key: str) -> str:
    return hashlib.sha1(f"{seed}|{key}".encode()).hexdigest()


def voice_key(rec: dict) -> tuple[str, str]:
    return (rec.get("corpus", "tata"), rec.get("speaker") or rec["idx"])


def group_of(rec: dict) -> str:
    return SPL.group_key(rec)


def usable(rec: dict, cfg: VoiceCfg) -> bool:
    spk = rec.get("speaker")
    if not spk or str(spk).endswith(UNASSIGNED_SUFFIX):
        return False
    if rec.get("corpus", "tata") in cfg.exclude_corpora:
        return False
    d = float(rec.get("dur_proc_s", 0.0))
    if not (cfg.min_dur_s <= d <= cfg.max_dur_s):
        return False
    return len((rec.get("text") or "").strip()) >= cfg.min_text_chars


def gender_of(rec: dict) -> str | None:
    """'F' | 'M' | None (campo `gender` do registro; aceita 'female_feminine', 'male', 'f', ...)."""
    g = str(rec.get("gender") or "").strip().lower()
    if g.startswith("female") or g == "f":
        return "F"
    if g.startswith("male") or g == "m":
        return "M"
    return None


def balance_gender(sel: list[dict], corpora, seed: int = 42) -> list[dict]:
    """Nos corpora dados: descarta vozes de genero desconhecido e mantem o MESMO numero de vozes F e M
    (as que sobram sao escolhidas por hash, deterministico). Outros corpora passam intactos."""
    corpora = set(corpora)
    by: dict[tuple, list[dict]] = defaultdict(list)
    for r in sel:
        by[voice_key(r)].append(r)
    keep: set = set()
    for c in corpora:
        vs = {"F": [], "M": []}
        for k, lst in by.items():
            if k[0] != c:
                continue
            g = gender_of(lst[0])
            if g:
                vs[g].append(k)
        n = min(len(vs["F"]), len(vs["M"]))
        for g in ("F", "M"):
            vs[g].sort(key=lambda k: _h(seed, f"gen|{k[0]}|{k[1]}"))
            keep.update(vs[g][:n])
    return [r for r in sel if r.get("corpus", "tata") not in corpora or voice_key(r) in keep]


def select_voices(recs: list[dict], cfg: VoiceCfg) -> list[dict]:
    """Registros mantidos: so vozes com >= min_clips clipes utilizaveis, cortadas em `cap` clipes
    (ordem por hash -> amostra aleatoria deterministica, sem favorecer o inicio do corpus)."""
    by: dict[tuple, list[dict]] = defaultdict(list)
    for r in recs:
        if usable(r, cfg):
            by[voice_key(r)].append(r)
    out: list[dict] = []
    for (corpus, spk), lst in sorted(by.items()):
        if len(lst) < cfg.min_clips:
            continue
        cap = int(cfg.caps.get(corpus, cfg.default_cap))
        lst = sorted(lst, key=lambda r: _h(cfg.seed, r["idx"]))[:cap]
        out.extend(sorted(lst, key=lambda r: r["idx"]))
    if cfg.balance_gender:
        out = balance_gender(out, cfg.balance_gender, cfg.seed)
    return out


def make_voice_splits(sel: list[dict], cfg: VoiceCfg) -> dict[str, list[str]]:
    """{train,val,test} -> [idx]. Grupos inteiros; nenhum grupo em >1 split (audita-se depois)."""
    groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in sel:
        groups[r.get("corpus", "tata")][group_of(r)].append(r)
    hold: dict[str, str] = {}                 # grupo -> split
    for corpus in sorted(groups):
        n_val, n_test = cfg.holdout_groups.get(corpus, (0, 0))
        el = [g for g, v in groups[corpus].items() if len(v) >= cfg.min_holdout_clips]
        el.sort(key=lambda g: _h(cfg.seed, f"hold|{g}"))
        if corpus in cfg.balance_gender:                 # val/test com F e M em partes iguais
            gen = {g: gender_of(groups[corpus][g][0]) for g in el}
            fem = [g for g in el if gen[g] == "F"]
            mal = [g for g in el if gen[g] == "M"]
            inter = [x for pair in zip(fem, mal) for x in pair]
            el = inter + [g for g in el if g not in set(inter)]
        # nunca esvazia o treino do corpus
        n_val = min(n_val, max(0, len(el) - 1))
        n_test = min(n_test, max(0, len(el) - n_val - 1))
        for g in el[:n_val]:
            hold[g] = "val"
        for g in el[n_val:n_val + n_test]:
            hold[g] = "test"
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    spk_n: dict[tuple, int] = defaultdict(int)
    for r in sorted(sel, key=lambda r: _h(cfg.seed, f"cap|{r['idx']}")):
        dest = hold.get(group_of(r), "train")
        if dest != "train":
            k = voice_key(r)
            if spk_n[k] >= cfg.heldout_cap:
                continue                      # excedente de val/test fica fora de tudo
            spk_n[k] += 1
        out[dest].append(r["idx"])
    return {k: sorted(v) for k, v in out.items()}


# ------------------------------------------------------------------ pesos por massa
def speaker_mass_weights(items: list[dict], corpus_share: dict | None = None, beta: float = 0.5,
                         cap_spk: float = 0.01, cap_grp: float = 0.02,
                         total_samples: int | None = None, max_repeat: float = 0.0,
                         max_iter: int = 500) -> tuple[dict[str, float], dict]:
    """idx -> probabilidade (soma 1) por CLIPE, com massa por VOZ <= cap_spk e por GRUPO <= cap_grp.

    items: registros (idx, corpus, speaker). Massa inicial por corpus = `corpus_share`
    (renormalizada); dentro do corpus, voz ∝ n_clipes**beta (beta=0 -> uniforme por voz).
    Os tetos sao impostos por preenchimento iterativo ("water-filling"): o excedente de uma voz/grupo
    no teto vai para as demais, proporcionalmente.

    `max_repeat` (>0, com `total_samples`): teto de repeticao — a voz nao pode receber mais que
    n_clipes * max_repeat amostras no treino todo (evita decorar clipes de vozes com poucos clipes).

    Se inviavel (soma dos tetos < 1), os tetos sao relaxados proporcionalmente (info['relaxed']).
    """
    share = dict(corpus_share or DEFAULT_CORPUS_SHARE)
    n_v: dict[tuple, int] = defaultdict(int)
    grp_of: dict[tuple, str] = {}
    for r in items:
        k = voice_key(r)
        n_v[k] += 1
        grp_of[k] = group_of(r)
    voices = sorted(n_v)
    corp_present = sorted({c for c, _ in voices})
    tot_share = sum(share.get(c, 1.0) for c in corp_present) or 1.0
    m: dict[tuple, float] = {}
    for c in corp_present:
        vs = [v for v in voices if v[0] == c]
        raw = {v: n_v[v] ** beta for v in vs}
        s = sum(raw.values())
        for v in vs:
            m[v] = (share.get(c, 1.0) / tot_share) * raw[v] / s
    cap_v = {v: cap_spk for v in voices}
    if max_repeat and total_samples:
        for v in voices:
            cap_v[v] = min(cap_v[v], n_v[v] * float(max_repeat) / float(total_samples))
    relaxed = False
    if sum(cap_v.values()) < 1.0:                           # inviavel: relaxa tudo na mesma proporcao
        f = 1.0 / sum(cap_v.values()) * 1.0001
        cap_v = {v: c * f for v, c in cap_v.items()}
        relaxed = True
    groups: dict[str, list[tuple]] = defaultdict(list)
    for v in voices:
        groups[grp_of[v]].append(v)
    cg = cap_grp
    if len(groups) * cg < 1.0:
        cg = 1.0 / len(groups) * 1.0001
        relaxed = True
    for _ in range(max_iter):
        for g, vs in groups.items():                       # teto por grupo
            tot = sum(m[v] for v in vs)
            if tot > cg:
                f = cg / tot
                for v in vs:
                    m[v] *= f
        for v in voices:                                   # teto por voz
            if m[v] > cap_v[v]:
                m[v] = cap_v[v]
        deficit = 1.0 - sum(m.values())
        if deficit < 1e-10:
            break
        gtot = {g: sum(m[v] for v in vs) for g, vs in groups.items()}
        free = {v: max(0.0, min(cap_v[v] - m[v], cg - gtot[grp_of[v]])) for v in voices}
        wts = {v: m[v] for v in voices if free[v] > 1e-12}
        sw = sum(wts.values())
        if sw <= 0:
            break
        for v, w in wts.items():
            m[v] += min(free[v], deficit * w / sw)
    tot = sum(m.values())
    m = {v: x / tot for v, x in m.items()}                 # normalizacao final (erro residual ~1e-10)
    w = {r["idx"]: m[voice_key(r)] / n_v[voice_key(r)] for r in items}
    gmass = {g: sum(m[v] for v in vs) for g, vs in groups.items()}
    rep = {v: (m[v] * total_samples / n_v[v]) if total_samples else float("nan") for v in voices}
    info = {"n_voices": len(voices), "n_groups": len(groups), "relaxed": relaxed,
            "max_voice_mass": max(m.values()), "max_group_mass": max(gmass.values()),
            "corpus_mass": {c: sum(m[v] for v in voices if v[0] == c) for c in corp_present},
            "eff_voices": 1.0 / sum(x * x for x in m.values()),
            "max_repeat_obs": max(rep.values()) if total_samples else float("nan")}
    return w, info


# ------------------------------------------------------------------ relatorio
def voice_table(sel: list[dict], splits: dict[str, list[str]]) -> list[dict]:
    where = {i: s for s, lst in splits.items() for i in lst}
    agg: dict[tuple, dict] = {}
    for r in sel:
        s = where.get(r["idx"])
        if s is None:
            continue
        k = (voice_key(r), s)
        d = agg.setdefault(k, {"corpus": r.get("corpus", "tata"), "speaker": voice_key(r)[1],
                               "group": group_of(r), "split": s, "n_clips": 0, "hours": 0.0})
        d["n_clips"] += 1
        d["hours"] += float(r.get("dur_proc_s", 0.0)) / 3600
    rows = sorted(agg.values(), key=lambda d: (d["split"], d["corpus"], d["group"], d["speaker"]))
    for d in rows:
        d["hours"] = round(d["hours"], 4)
    return rows


def split_summary(sel: list[dict], splits: dict[str, list[str]]) -> dict:
    by_idx = {r["idx"]: r for r in sel}
    out: dict = {}
    for s, lst in splits.items():
        per: dict = {}
        for i in lst:
            r = by_idx[i]
            d = per.setdefault(r.get("corpus", "tata"), {"clips": 0, "hours": 0.0, "voices": set(),
                                                          "groups": set()})
            d["clips"] += 1
            d["hours"] += float(r.get("dur_proc_s", 0.0)) / 3600
            d["voices"].add(voice_key(r))
            d["groups"].add(group_of(r))
        out[s] = {c: {"clips": d["clips"], "hours": round(d["hours"], 2), "voices": len(d["voices"]),
                      "groups": len(d["groups"])} for c, d in sorted(per.items())}
        out[s]["_total"] = {"clips": len(lst),
                            "hours": round(sum(v["hours"] for v in out[s].values()), 2),
                            "voices": len({voice_key(by_idx[i]) for i in lst})}
    return out


def audit(sel: list[dict], splits: dict[str, list[str]]) -> dict:
    return SPL.audit_leaks(sel, splits)


def pick_eval_voices(recs: dict[str, dict], idxs: list[str], n: int = 4,
                     corpus_order=("tagarela", "podcast", "cml_pt", "cetuc", "cv_pt", "tata"),
                     min_clips: int = 6, min_prompt_s: float = 4.0, max_prompt_s: float = 9.0,
                     seed: int = 7) -> list[dict]:
    """Vozes (de um split NAO treinado) para a avaliacao periodica de SECS/WER.

    Devolve [{name, corpus, speaker, prompt_idx, heldout_idx:[...]}]; uma voz por corpus (na ordem
    `corpus_order`, ciclando se n > corpora) -> cobre dominios diferentes. Deterministico.
    """
    by: dict[tuple, list[dict]] = defaultdict(list)
    for i in idxs:
        r = recs.get(i)
        if r is None:
            continue
        spk = r.get("speaker")
        if not spk or str(spk).endswith(UNASSIGNED_SUFFIX):
            continue
        by[voice_key(r)].append(r)
    cand: dict[str, list[tuple]] = defaultdict(list)
    for k, lst in by.items():
        if len(lst) >= min_clips and any(min_prompt_s <= float(r["dur_proc_s"]) <= max_prompt_s
                                         for r in lst):
            cand[k[0]].append(k)
    for c in cand:
        cand[c].sort(key=lambda k: _h(seed, f"{k[0]}|{k[1]}"))
    out: list[dict] = []
    used: set = set()
    rounds = 0
    while len(out) < n and rounds < n + len(corpus_order):
        for c in corpus_order:
            if len(out) >= n:
                break
            lst = [k for k in cand.get(c, []) if k not in used]
            if not lst:
                continue
            k = lst[0]
            used.add(k)
            clips = sorted(by[k], key=lambda r: r["idx"])
            prompt = next(r for r in clips if min_prompt_s <= float(r["dur_proc_s"]) <= max_prompt_s)
            rest = [r for r in clips if r is not prompt and float(r["dur_proc_s"]) >= 3.0]
            step = max(1, len(rest) // 5)
            ho = rest[::step][:5]
            out.append({"name": f"{c}_{len(out)}", "corpus": c, "speaker": k[1],
                        "prompt_idx": prompt["idx"], "heldout_idx": [r["idx"] for r in ho]})
        rounds += 1
    return out


def ceiling_floor(ceil_vals: list[float], floor_vals: list[float]) -> tuple[float, float]:
    c = sum(ceil_vals) / len(ceil_vals) if ceil_vals else float("nan")
    f = sum(floor_vals) / len(floor_vals) if floor_vals else float("nan")
    return c, f


def normalized_secs(secs: float, ceiling: float, floor: float) -> float:
    d = ceiling - floor
    if not (d > 1e-6) or math.isnan(d):
        return float("nan")
    return (secs - floor) / d
