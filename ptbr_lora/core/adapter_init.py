"""Warm start: carrega os pesos de um adapter LoRA (ex.: so atencao) num LoRA com MAIS alvos (ex.: attn + MLP).

Uso (train_lora.py --init-from-adapter): o modelo novo e criado com `--targets all`; os modulos que existem no adapter
antigo recebem os pesos dele; os modulos novos (MLP) ficam com a inicializacao padrao do LoRA (B = 0), entao a saida no
passo 0 e IDENTICA a do adapter antigo. Tudo e conferido tensor a tensor (nao depende do retorno do peft).
"""
from __future__ import annotations

import json
from pathlib import Path



def _read_cfg(adapter_dir: Path) -> dict:
    return json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))


def _model_key(saved_key: str, adapter_name: str = "default") -> str:
    # 'base_model.model.X.lora_A.weight' -> 'base_model.model.X.lora_A.default.weight'
    head, tail = saved_key.rsplit(".weight", 1)[0], "weight"
    return f"{head}.{adapter_name}.{tail}"


def check_compat(adapter_dir, *, rank: int, alpha: int, use_rslora: bool) -> dict:
    """Falha cedo se rank/alpha/rsLoRA do adapter de origem diferem do que sera treinado (a escala mudaria)."""
    cfg = _read_cfg(Path(adapter_dir))
    got = (int(cfg["r"]), float(cfg["lora_alpha"]), bool(cfg.get("use_rslora", False)))
    want = (int(rank), float(alpha), bool(use_rslora))
    if got != want:
        raise ValueError(
            f"adapter de origem (r, alpha, rslora)={got} difere do pedido {want}: a escala efetiva mudaria "
            f"e o passo 0 deixaria de ser igual ao adapter de origem")
    return cfg


def load_into(pm, adapter_dir, *, adapter_name: str = "default") -> dict:
    """Copia os tensores do adapter para `pm` (PeftModel). Retorna um relatorio e levanta se algo nao casar."""
    import torch
    from safetensors.torch import load_file

    adapter_dir = Path(adapter_dir)
    st = adapter_dir / "adapter_model.safetensors"
    if not st.is_file():
        raise FileNotFoundError(f"sem adapter_model.safetensors em {adapter_dir}")
    saved = load_file(str(st))
    params = dict(pm.named_parameters())
    missing_in_model = [k for k in saved if _model_key(k, adapter_name) not in params]
    if missing_in_model:
        raise KeyError(f"{len(missing_in_model)} tensores do adapter nao existem no modelo novo "
                       f"(ex.: {missing_in_model[:3]}). Os alvos do modelo novo precisam incluir os do adapter.")
    with torch.no_grad():
        for k, v in saved.items():
            p = params[_model_key(k, adapter_name)]
            if tuple(p.shape) != tuple(v.shape):
                raise ValueError(f"forma diferente em {k}: adapter {tuple(v.shape)} x modelo {tuple(p.shape)}")
            p.copy_(v.to(device=p.device, dtype=p.dtype))
    # conferencia independente
    n_diff = 0
    with torch.no_grad():
        for k, v in saved.items():
            p = params[_model_key(k, adapter_name)]
            if not torch.equal(p.detach().cpu().to(torch.float32), v.to(torch.float32)):
                # copiado de F32 -> dtype do modelo: aceita so se a diferenca for de arredondamento do proprio dtype
                if not torch.allclose(p.detach().cpu().to(torch.float32), v.to(torch.float32), rtol=1e-2, atol=1e-3):
                    n_diff += 1
    if n_diff:
        raise RuntimeError(f"{n_diff} tensores nao batem apos a copia")
    saved_names = {_model_key(k, adapter_name) for k in saved}
    new_b_nonzero = []
    n_new = 0
    with torch.no_grad():
        for name, p in params.items():
            if ".lora_" in name and name not in saved_names:
                n_new += 1
                if ".lora_B." in name and float(p.detach().abs().max()) != 0.0:
                    new_b_nonzero.append(name)
    if new_b_nonzero:
        raise RuntimeError(f"modulos novos com lora_B != 0 (a saida do passo 0 nao seria igual): {new_b_nonzero[:3]}")
    return {"loaded_tensors": len(saved), "new_lora_tensors": n_new, "new_lora_B_all_zero": True}
