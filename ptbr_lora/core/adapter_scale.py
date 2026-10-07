"""adapter_scale.py — multiplica a escala TREINADA dos modulos LoRA (PEFT). Modulo puro.

Corrige o cache antigo, que guardava o scaling-base so das chaves presentes na 1a chamada:
um adapter carregado depois (nome novo) ficava com a escala errada/acumulada. Aqui o base
e registrado POR CHAVE, no momento em que a chave aparece pela 1a vez (ainda intacta).

Compartilhado: repo de treino (`ptbr_lora/core`) e UI (`core/`, via scripts/sync_shared.py).
"""
from __future__ import annotations

_ATTR = "_ptbr_base_scaling"


def apply_adapter_scale(model, factor: float, adapter: str | None = None) -> int:
    """Escala = escala_treinada * factor (1.0 = como treinado). Idempotente.

    adapter=None -> todos os adapters; senao so a chave `adapter`.
    Retorna o numero de (modulo, adapter) ajustados.
    """
    n = 0
    f = float(factor)
    for m in model.modules():
        sc = getattr(m, "scaling", None)
        if not (isinstance(sc, dict) and sc):
            continue
        base = m.__dict__.get(_ATTR)
        if base is None:
            base = {}
            try:
                object.__setattr__(m, _ATTR, base)
            except Exception:  # noqa: BLE001
                m.__dict__[_ATTR] = base
        for k in list(sc.keys()):
            if k not in base:
                base[k] = float(sc[k])            # 1a vez que vemos a chave: ainda nao mexida
            if adapter is not None and k != adapter:
                continue
            sc[k] = base[k] * f
            n += 1
    return n


def reset_adapter_scale(model) -> int:
    return apply_adapter_scale(model, 1.0)


def trained_scales(model) -> dict[str, float]:
    """Escala treinada (base) por adapter — a primeira encontrada; util para log/UI."""
    out: dict[str, float] = {}
    for m in model.modules():
        sc = getattr(m, "scaling", None)
        if isinstance(sc, dict):
            base = m.__dict__.get(_ATTR, {})
            for k, v in sc.items():
                out.setdefault(k, base.get(k, float(v)))
    return out
