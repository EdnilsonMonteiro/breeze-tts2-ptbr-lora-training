"""gen_core.py — nucleo de geracao (carregar modelo + gerar 1 amostra) para as tools.

Extraido do fluxo de sweep para que `gerar_em_blocos.py` (protocolo de producao) nao
dependa de scripts de teste. Reutiliza o engine do fork via `common_breeze`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]          # raiz do repo (engine + ptbr_lora)
_CORE = Path(__file__).resolve().parents[1] / "core"
for _p in (str(_CORE), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import common_breeze as CB  # noqa: E402

DEFAULT_ADAPTER = str(
    Path(CB.TRAINING) / "runs" / "r64_03" / "checkpoints" / "step4000"
)


def apply_adapter_scale(model, scale: float) -> int:
    """Seta a escala do adapter LoRA (0.3-1.0 atenua over-steering). Retorna n modulos."""
    n = 0
    for m in model.modules():
        sc = getattr(m, "scaling", None)
        if isinstance(sc, dict):
            for k in list(sc):
                sc[k] = float(scale)
                n += 1
    return n


def load_model(adapter: str, device: str):
    from peft import PeftModel

    raw = CB.load_breeze_model(device, attn="eager")
    if adapter:
        raw = PeftModel.from_pretrained(raw, adapter)
    raw.eval()
    tok = CB.load_text_tokenizer()
    atok = CB.load_audio_tokenizer(device)
    return raw, tok, atok


def generate_one(model, tok, atok, cfg: dict, seed: int, ref_audio, ref_text: str,
                 text: str, device: str = "cuda") -> tuple[np.ndarray, int]:
    import torch

    from breeze_infer.runtime import set_all_seeds, update_generation_config_for_breeze
    from breeze_infer.templates import get_template, prepare_inputs

    update_generation_config_for_breeze(model)
    template = get_template(cfg["template"])
    request = {"id": f"seed{seed}", "text": text, "instruction": cfg["instruction"],
               "speaker": cfg.get("speaker", "S0")}
    if ref_audio:
        request["ref_audio_path"] = str(ref_audio)
        request["ref_text"] = ref_text
    set_all_seeds(int(seed))
    inputs = prepare_inputs(
        tok, atok, model, [request], template,
        guidance_scale=float(cfg["cfg_scale"]),
        guidance_scale_ref=float(cfg["cfg_ref"]) if cfg.get("use_dual_cfg") else None,
        guidance_scale_ins=float(cfg["cfg_ins"]) if cfg.get("use_dual_cfg") else None,
    )
    with torch.inference_mode():
        res = model.generate(
            **inputs, output_audio=True, audio_tokenizer=atok,
            max_new_tokens=int(cfg["max_new_tokens"]), do_sample=True,
            temperature=float(cfg["temperature"]), top_k=int(cfg["top_k"]),
            top_p=float(cfg["top_p"]),
        )
    wav_t = res[0] if isinstance(res, (list, tuple)) else getattr(res, "audio", [None])[0]
    while wav_t.dim() > 1:
        wav_t = wav_t[0]
    return wav_t.detach().float().cpu().numpy(), atok.get_output_sample_rate()
