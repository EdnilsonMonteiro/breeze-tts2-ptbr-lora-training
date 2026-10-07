"""refs.py — escolha de referencia de locutor e da CONDICAO de cada exemplo. Modulo puro.

Protocolo novo (corrige a auditoria 2026-09):
  * a referencia e OUTRO clipe do mesmo locutor, sorteado por (epoca, idx) — nada de
    "referencia canonica" fixa (o modelo memorizava 1 clipe por locutor) nem self-ref
    (o alvo vazava para o prompt; o treino vira copia);
  * o pool de referencias vem do MESMO split do item (val usa val, treino usa treino):
    o val mede exatamente a condicao de uso (locutor nao visto, clipe de referencia
    diferente do alvo);
  * buckets `:99` (locutores dissolvidos / nao atribuidos) NAO sao locutor: nunca viram
    pool de referencia; esses itens caem em condicoes sem referencia.
"""
from __future__ import annotations

import hashlib

MAX_REF_S = 10.2
MIN_REF_S = 2.5
UNASSIGNED_SUFFIX = ":99"

REF_VARIANTS = ("ref_edit_tata", "ref_clone_tata")
NO_REF_VARIANTS = ("tts_instruction", "tts_plain")
VARIANTS = REF_VARIANTS + NO_REF_VARIANTS

DEFAULT_MIX = {"ref_edit_tata": 0.75, "ref_clone_tata": 0.10,
               "tts_instruction": 0.10, "tts_plain": 0.05}

_ALIASES = {"ref_edit": "ref_edit_tata", "ref_clone": "ref_clone_tata",
            "instruction": "tts_instruction", "plain": "tts_plain",
            "ref_edit_auto": "ref_edit_tata"}


def parse_mix(spec: str | dict | None) -> dict[str, float]:
    """'ref_edit=0.75,ref_clone=0.1,instruction=0.1,plain=0.05' -> mix normalizado."""
    if spec is None:
        return dict(DEFAULT_MIX)
    if isinstance(spec, str):
        items = {}
        for part in spec.split(","):
            if part.strip():
                k, v = part.split("=")
                items[k.strip()] = float(v)
    else:
        items = dict(spec)
    mix: dict[str, float] = {}
    for k, v in items.items():
        k = _ALIASES.get(k, k)
        if k not in VARIANTS:
            raise ValueError(f"variante desconhecida no mix: {k!r} (validas: {VARIANTS})")
        mix[k] = mix.get(k, 0.0) + float(v)
    tot = sum(mix.values())
    if tot <= 0:
        raise ValueError("mix vazio")
    return {k: v / tot for k, v in mix.items() if v > 0}


def mix_from_ref_edit_frac(frac: float) -> dict[str, float]:
    """Compatibilidade com --ref-edit-frac: fracao com ref_edit; o resto e tts_instruction."""
    frac = min(1.0, max(0.0, float(frac)))
    return {k: v for k, v in {"ref_edit_tata": frac, "tts_instruction": 1.0 - frac}.items() if v > 0}


def restrict_to_ref(mix: dict[str, float]) -> dict[str, float]:
    """Mix so com variantes que usam referencia (val cross-ref)."""
    r = {k: v for k, v in mix.items() if k in REF_VARIANTS}
    if not r:
        r = {"ref_edit_tata": 1.0}
    tot = sum(r.values())
    return {k: v / tot for k, v in r.items()}


def eligible_ref_speaker(spk: str | None) -> bool:
    return bool(spk) and not str(spk).endswith(UNASSIGNED_SUFFIX)


def ref_duration_ok(rec: dict) -> bool:
    d = float(rec.get("dur_proc_s", 0.0))
    return MIN_REF_S <= d <= MAX_REF_S


def build_pools(recs: list[dict], idxs) -> dict[str, list[str]]:
    """speaker -> [idx] (ordenado), so com idxs do split, locutor elegivel e clipe adequado."""
    keep = set(idxs)
    pools: dict[str, list[str]] = {}
    for r in recs:
        if r["idx"] not in keep:
            continue
        spk = r.get("speaker")
        if eligible_ref_speaker(spk) and ref_duration_ok(r):
            pools.setdefault(spk, []).append(r["idx"])
    for v in pools.values():
        v.sort()
    return pools


def _u(*parts) -> float:
    h = hashlib.sha1("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def pick_ref(rec: dict, pools: dict[str, list[str]], epoch: int = 0, seed: int = 0) -> str | None:
    """idx de OUTRO clipe do mesmo locutor (None se nao ha)."""
    spk = rec.get("speaker")
    if not eligible_ref_speaker(spk):
        return None
    pool = [i for i in pools.get(spk, ()) if i != rec["idx"]]
    if not pool:
        return None
    return pool[int(_u("ref", seed, epoch, rec["idx"]) * len(pool)) % len(pool)]


def sample_variant(idx: str, mix: dict[str, float], epoch: int = 0, seed: int = 0) -> str:
    x = _u("var", seed, epoch, idx)
    acc = 0.0
    items = sorted(mix.items())
    for k, p in items:
        acc += p
        if x < acc:
            return k
    return items[-1][0]


def resolve_condition(rec: dict, pools: dict[str, list[str]], mix: dict[str, float],
                      epoch: int = 0, seed: int = 0,
                      require_ref: bool = False) -> tuple[str, str | None] | None:
    """(variante, ref_idx|None) para o item. Sem referencia disponivel, variantes com ref
    caem para `tts_instruction`; com `require_ref=True` (val cross-ref) devolve None."""
    v = sample_variant(rec["idx"], mix, epoch, seed)
    if v in REF_VARIANTS:
        ref = pick_ref(rec, pools, epoch, seed)
        if ref is not None:
            return v, ref
        return None if require_ref else ("tts_instruction", None)
    return None if require_ref else (v, None)
