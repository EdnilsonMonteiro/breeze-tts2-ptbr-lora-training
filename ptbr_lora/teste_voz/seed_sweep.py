"""seed_sweep.py — gera a MESMA frase para varias seeds (e/ou varias configs).

Carrega base + adapter UMA vez e gera N seeds por config, salvando WAVs e um
manifesto. Resumivel: pula WAVs ja existentes.

Exemplos:
  # 50 seeds na config padrao
  python seed_sweep.py --seeds 1-50 --out-dir teste_seeds

  # mini-teste de estabilidade (temp 0.7, dual-CFG ref=3)
  python seed_sweep.py --seeds 1-10 --temperature 0.7 --use-dual-cfg --cfg-ref 3 --out-dir teste_seeds

  # grade de configs via JSON
  python seed_sweep.py --seeds 1-5 --configs grid.json --out-dir teste_seeds
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

_ROOT = Path(__file__).resolve().parents[2]          # raiz do repo (engine + ptbr_lora)
_CORE = Path(__file__).resolve().parents[1] / "core"
for _p in (str(_CORE), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import common_breeze as CB  # noqa: E402

DEFAULT_ADAPTER = str(
    Path(CB.TRAINING) / "runs" / "r64_03" / "checkpoints" / "epoch0_val6_224"
)
HERE = Path(__file__).resolve().parent


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def tag_for(cfg: dict) -> str:
    t = f"T{cfg['temperature']:g}_k{cfg['top_k']}_p{cfg['top_p']:g}_cfg{cfg['cfg_scale']:g}"
    if cfg.get("use_dual_cfg"):
        t += f"_dual{cfg.get('cfg_ref', 1.0):g}-{cfg.get('cfg_ins', 1.0):g}"
    if cfg.get("template") != "ref_edit_tata":
        t += f"_{cfg['template']}"
    return t


def load_model(adapter: str, device: str):
    import torch
    from peft import PeftModel

    raw = CB.load_breeze_model(device, attn="eager")
    if adapter:
        raw = PeftModel.from_pretrained(raw, adapter)
    raw.eval()
    tok = CB.load_text_tokenizer()
    atok = CB.load_audio_tokenizer(device)
    return raw, tok, atok


def generate_one(model, tok, atok, cfg, seed, ref_audio, ref_text, text, device) -> np.ndarray:
    import torch

    from breeze_infer.runtime import set_all_seeds, update_generation_config_for_breeze
    from breeze_infer.templates import get_template, prepare_inputs

    update_generation_config_for_breeze(model)
    template = get_template(cfg["template"])
    request = {"id": f"seed{seed}", "text": text, "instruction": cfg["instruction"],
               "speaker": "S0"}
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep de seeds/configs de TTS.")
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER, help="pasta do adapter ou id do HF")
    ap.add_argument("--ref-audio", default=str(HERE / "VozEdnilson.wav"))
    ap.add_argument("--ref-text-file", default=str(HERE / "ref_VozEdnilson.txt"))
    ap.add_argument("--text-file", default=str(HERE / "frase_alvo.txt"))
    ap.add_argument("--instruction", default="Fale de forma clara e natural.")
    ap.add_argument("--template", default="ref_edit_tata",
                    choices=["ref_edit_tata", "ref_clone_tata", "tts_instruction"])
    ap.add_argument("--seeds", default="1-50")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--use-dual-cfg", action="store_true")
    ap.add_argument("--cfg-ref", type=float, default=3.0)
    ap.add_argument("--cfg-ins", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=1200)
    ap.add_argument("--out-dir", default=str(HERE / "teste_seeds"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--configs", default=None, help="JSON com lista de configs (opcional)")
    ap.add_argument("--tag", default=None, help="tag explicita (override)")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    text = Path(args.text_file).read_text(encoding="utf-8").strip()
    ref_text = ""
    ref_audio = None
    if args.template != "tts_instruction":
        ref_audio = Path(args.ref_audio)
        ref_text = Path(args.ref_text_file).read_text(encoding="utf-8").strip()

    base_cfg = dict(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                    cfg_scale=args.cfg_scale, use_dual_cfg=args.use_dual_cfg,
                    cfg_ref=args.cfg_ref, cfg_ins=args.cfg_ins,
                    template=args.template, instruction=args.instruction,
                    max_new_tokens=args.max_new_tokens)
    if args.configs:
        cfgs = json.loads(Path(args.configs).read_text(encoding="utf-8"))
    else:
        cfgs = [base_cfg]

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = out_root / "manifest.jsonl"

    if args.adapter and not Path(args.adapter).is_dir():
        dl = CB.ensure_adapter(repo_id=args.adapter) if hasattr(CB, "ensure_adapter") else None
        if dl:
            args.adapter = str(dl)

    print(f"[sweep] adapter={args.adapter}")
    print(f"[sweep] ref={ref_audio} ({len(ref_text)} chars) | seeds={len(seeds)} | cfgs={len(cfgs)}")
    model, tok, atok = load_model(args.adapter, args.device)

    for cfg in cfgs:
        cfg = {**base_cfg, **cfg}
        tag = args.tag or tag_for(cfg)
        cdir = out_root / tag
        cdir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            wav_p = cdir / f"seed_{seed:03d}.wav"
            if wav_p.exists():
                print(f"[sweep] {tag} seed {seed}: existe, pulando", flush=True)
                continue
            t0 = time.time()
            wav, sr = generate_one(model, tok, atok, cfg, seed, ref_audio, ref_text,
                                   text, args.device)
            sf.write(str(wav_p), np.clip(wav, -1.0, 1.0), sr, subtype="PCM_16")
            dur = len(wav) / sr
            rec = {"tag": tag, "seed": seed, "wav": str(wav_p), "sr": sr,
                   "gen_dur_s": round(dur, 2), "gen_time_s": round(time.time() - t0, 1),
                   **cfg}
            with manifest.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[sweep] {tag} seed {seed}: {dur:.1f}s em {time.time()-t0:.0f}s -> {wav_p.name}",
                  flush=True)
    print("[sweep] FIM")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
