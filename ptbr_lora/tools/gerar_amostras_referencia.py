"""gerar_amostras_referencia.py — faz a VOZ DE REFERENCIA falar as frases de avaliacao.

Usa o template ref_edit_tata (clone + instrucao) com a referencia `VozEdnilson` e
gera, para cada frase de SAMPLE_TEXTS, um WAV. Serve como "baseline de referencia"
para comparar com as amostras dos checkpoints (rodando com --adapter vazio = base,
ou com um checkpoint = adapter).

Uso:
  python ptbr_lora/tools/gerar_amostras_referencia.py \
    --ref-audio "C:\\IA\\Breeze-tts\\voices\\VozEdnilson.wav" \
    --out "C:\\IA\\Breeze-tts\\training\\runs\\r70_01\\reference"

Sem --adapter, escolhe o checkpoint mais recente de runs/r70_01/checkpoints; se nao
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

SAMPLE_TEXTS = [
    ("ola-pt", "Olá! Este é um teste de voz em português brasileiro."),
    ("numeros-pt", "O número da minha casa é quinze oh dois, no bairro Jardim Europa."),
    ("clima-pt", "A previsão do tempo indica pancadas de chuva à tarde, com temperaturas "
                 "entre dezesseis e vinte e três graus."),
    ("siglas-pt", "Atenção: CPF um dois três ponto quatro cinco seis ponto sete oito nove, traço zero um."),
    ("afetivo-pt", "Que saudade daquele café quentinho da vovó no fim da tarde!"),
    ("regressao-en", "The weather today is sunny with a gentle breeze from the east."),
    ("placa-pt", "O carro de placa ABC um D vinte e três foi apreendido ontem à noite."),
    ("letras-pt", "As vogais são A, E, I, O, U; e as consoantes seguem o alfabeto."),
    ("siglas2-pt", "O IBGE e o INSS divulgaram os números na quinta-feira passada."),
    ("oov-pt", "O buzinaço assustou o gatíneo enquanto ele papeava na varanda."),
    ("trabalenguas-pt", "O rato roeu a roupa do rei de Roma e o mundo se admirou."),
]


def _latest_checkpoint(run: str) -> str:
    d = CB.TRAINING / "runs" / run / "checkpoints"
    if not d.is_dir():
        return ""
    cks = [p for p in d.iterdir() if p.is_dir() and (p / "adapter_config.json").is_file()]
    if not cks:
        return ""
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
                    default=str(CB.ARTIFACTS / "voices" / "VozEdnilson" / "ref.wav"))
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--adapter", default=None, help="pasta do adapter; '' = base; default = checkpoint mais recente")
    ap.add_argument("--run", default="r70_01", help="run usado para achar o checkpoint padrao")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--instruction", default="Fale com clareza e naturalidade.")
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
    tag = "base" if not adapter else Path(adapter).name
    out = Path(args.out) if args.out else (CB.TRAINING / "runs" / args.run / "reference")
    out.mkdir(parents=True, exist_ok=True)

    # guarda a referencia original para audicao
    try:
        shutil.copy(str(ref), str(out / "00_referencia_original.wav"))
    except Exception:  # noqa: BLE001
        pass

    print(f"[ref] referencia={ref}")
    print(f"[ref] adapter={adapter or '(base)'}  device={args.device}  out={out}")

    model, tok, atok = gen_core.load_model(adapter, args.device)
    cfg = {
        "template": "ref_edit_tata", "instruction": args.instruction,
        "cfg_scale": 1.0, "use_dual_cfg": False, "cfg_ref": 1.0, "cfg_ins": 1.0,
        "temperature": args.temperature, "top_k": 50, "top_p": 1.0,
        "max_new_tokens": 1200, "speaker": "S0",
    }
    for i, (slug, text) in enumerate(SAMPLE_TEXTS):
        p = out / f"{i:02d}_{slug}_{tag}.wav"
        if p.exists():
            print(f"[ref] {p.name}: existe, pulando")
            continue
        t0 = time.time()
        wav, sr = gen_core.generate_one(
            model, tok, atok, cfg, 42, ref, ref_text, text_norm.normalize(text), args.device)
        sf.write(str(p), np.clip(wav, -1.0, 1.0), int(sr), subtype="PCM_16")
        print(f"[ref] {p.name}: {len(wav)/sr:.1f}s em {time.time()-t0:.0f}s -> {p}", flush=True)
    print("[ref] FIM")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
