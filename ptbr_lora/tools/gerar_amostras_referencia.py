"""gerar_amostras_referencia.py — faz a VOZ DE REFERENCIA falar as frases de avaliacao.

Usa o template ref_edit_tata (clone + instrucao) com a referencia `voz_autor` e
gera, para cada frase de SAMPLE_TEXTS, um WAV. Serve como "baseline de referencia"
para comparar com as amostras dos checkpoints (rodando com --adapter vazio = base,
ou com um checkpoint = adapter).

Uso:
  python ptbr_lora/tools/gerar_amostras_referencia.py \
    --ref-audio "<ARTIFACTS>\\voices\\voz_autor.wav" \
    --out "<ARTIFACTS>\\training\\runs\\r72_01\\reference"

Sem --adapter, escolhe o checkpoint mais recente de runs/r72_01/checkpoints; se nao
houver, usa o modelo BASE (--adapter "").
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

_ROOT = Path(__file__).resolve().parents[2]
_CORE = Path(__file__).resolve().parents[1] / "core"
for _p in (str(_CORE), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import common_breeze as CB  # noqa: E402
import gen_core  # noqa: E402
import text_norm  # noqa: E402

from sample_texts import SAMPLE_TEXTS  # noqa: E402  (fonte unica, com o treino)
import reference_prep as RP  # noqa: E402


def _latest_checkpoint(run: str) -> str:
    d = CB.TRAINING / "runs" / run / "checkpoints"
    if not d.is_dir():
        return ""
    cks = [p for p in d.iterdir() if p.is_dir() and (p / "adapter_config.json").is_file()]
    if not cks:
        return ""
    for pref in ("final", "best"):                     # prefere o final/best ao mais recente
        for p in cks:
            if p.name == pref:
                return str(p)
    return str(max(cks, key=lambda p: p.stat().st_mtime))


def _ref_text_from_voices(ref_audio: Path) -> str:
    vj = CB.ARTIFACTS / "voices.json"
    if vj.is_file():
        try:
            data = json.loads(vj.read_text(encoding="utf-8"))
            for v in data.values():
                if v.get("ref_path") and Path(v["ref_path"]).name == ref_audio.name:
                    return (v.get("ref_text") or "").strip()
        except Exception:  # noqa: BLE001
            pass
    txt = ref_audio.with_suffix(".txt")
    return txt.read_text(encoding="utf-8").strip() if txt.is_file() else ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Gera as frases de referencia com a voz clone.")
    ap.add_argument("--ref-audio",
                    default=str(CB.ARTIFACTS / "voices" / "voz_autor" / "ref.wav"))
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--adapter", default=None, help="pasta do adapter; '' = base; default = checkpoint mais recente")
    ap.add_argument("--run", default="r72_01", help="run usado para achar o checkpoint padrao")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--instruction", default="Fale com clareza e naturalidade.")
    ap.add_argument("--candidates", type=int, default=1,
                    help="N seeds por frase; grava o MEDOID (median-of-N). Reduz a "
                         "variancia da amostra unica que distorce o SECS")
    ap.add_argument("--tag", default=None, help="forca o sufixo do arquivo (default = nome do adapter)")
    args = ap.parse_args()

    ref = Path(args.ref_audio)
    if not ref.is_file():
        sys.exit(f"[ref] referencia nao encontrada: {ref}")
    ref_text = (args.ref_text or _ref_text_from_voices(ref)).strip()
    if not ref_text:
        sys.exit("[ref] sem transcricao da referencia (use --ref-text)")

    if args.adapter is None:
        adapter = _latest_checkpoint(args.run)
    else:
        adapter = args.adapter
    tag = args.tag or ("base" if not adapter else Path(adapter).name)
    out = Path(args.out) if args.out else (CB.TRAINING / "runs" / args.run / "reference")
    out.mkdir(parents=True, exist_ok=True)

    # guarda a referencia original para audicao
    try:
        shutil.copy(str(ref), str(out / "00_referencia_original.wav"))
    except Exception:  # noqa: BLE001
        pass

    ref, _dur = RP.prepare_reference(ref, out / "_ref", "train")   # formato do treino
    ref_text = text_norm.normalize(ref_text)
    print(f"[ref] referencia={ref} ({_dur:.1f}s)")
    print(f"[ref] adapter={adapter or '(base)'}  device={args.device}  out={out}")

    model, tok, atok = gen_core.load_model(adapter, args.device)
    cfg = {
        "template": "ref_edit_tata", "instruction": args.instruction,
        "cfg_scale": 1.0, "use_dual_cfg": False, "cfg_ref": 1.0, "cfg_ins": 1.0,
        "temperature": args.temperature, "top_k": 50, "top_p": 1.0,
        "max_new_tokens": 400, "speaker": "S0",
    }
    n_cand = max(1, int(args.candidates))
    cand_dir = out / "_candidates"
    for i, (slug, text) in enumerate(SAMPLE_TEXTS):
        p = out / f"{i:02d}_{slug}_{tag}.wav"
        if p.exists():
            print(f"[ref] {p.name}: existe, pulando")
            continue
        t0 = time.time()
        norm_text = text_norm.normalize(text)
        if n_cand == 1:
            wav, sr = gen_core.generate_one(
                model, tok, atok, cfg, 42, ref, ref_text, norm_text, args.device)
            sf.write(str(p), np.clip(wav, -1.0, 1.0), int(sr), subtype="PCM_16")
            print(f"[ref] {p.name}: {len(wav)/sr:.1f}s em {time.time()-t0:.0f}s -> {p}",
                  flush=True)
            continue
        # protocolo medoid: gera N seeds, escolhe a geracao mais CENTRAL
        cand_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for k in range(n_cand):
            wav, sr = gen_core.generate_one(
                model, tok, atok, cfg, 42 + k, ref, ref_text, norm_text, args.device)
            cp = cand_dir / f"{i:02d}_{slug}_s{42+k:03d}.wav"
            sf.write(str(cp), np.clip(wav, -1.0, 1.0), int(sr), subtype="PCM_16")
            paths.append(cp)
        import metrics  # ECAPA (usa CPU por padrao p/ nao disputar a GPU)

        E = np.stack([metrics.embed(c, "cpu") for c in paths])
        center = E.mean(axis=0)
        scores = [metrics.cos(E[j], center) for j in range(len(paths))]
        best = int(np.argmax(scores))
        shutil.copy(str(paths[best]), str(p))
        durs = [round(float(sf.info(str(c)).duration), 1) for c in paths]
        print(f"[ref] {p.name}: medoid cand#{best}/n={n_cand} durs={durs} "
              f"em {time.time()-t0:.0f}s -> {p}", flush=True)
    print("[ref] FIM")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
