"""gerar_em_blocos.py — geração de voz em BLOCOS de <= 8-10 s, com escolha por medoid.

Por que blocos: o treino usa clipes de ~6 s (máx. 10 s) e a identidade degrada dentro
da própria geração (~0,013 de cos por segundo extra). Gerar 15-30 s num único passe é
extrapolação de comprimento — a voz "escorrega" do meio para o fim. Aqui o texto é
fatiado em blocos de <= --max-block-s (default 10 s, estimando ~2,6 palavras/s) e cada
bloco é gerado com a MESMA referência.

Protocolo por bloco:
  1. gera N candidatos com seeds distintas;
  2. mede em cada candidato: cos_ref_multi (média do cos ECAPA contra TODAS as
     referências do locutor), WER vs texto do bloco, drift (cos início - cos fim)
     e duração;
  3. filtra por gate (WER + duração relativa ao lote + duração plausível para o texto)
     e escolhe o MEDOID (candidato mais central), não o máximo do cos — o máximo é
     ruído de amostragem (seção 3.1 do doc);
  4. iguala o RMS dos blocos e concatena com pausas.

Uso:
  python gerar_em_blocos.py --voice VozEdnilsonA \
      --workspace "C:/IA/Breeze-tts/voices" \
      --adapter "C:/IA/Breeze-tts/training/runs/r64_03/checkpoints/step4000" \
      --text-file meu_texto.txt --candidates 8 --out "C:/IA/Breeze-tts/training/ui_out/exp1"

  # blocos menores (mais fidelidade, mais tempo de GPU):
  --max-block-s 8

  # clone puro (só identidade) vs voice direction (com instrução):
  --template ref_clone_tata           (default; cai p/ ref_edit_tata se o engine não tiver)
  --template ref_edit_tata --cfg-scale 4.0
  --use-dual-cfg --cfg-ref 1.2
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "core"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import metrics  # noqa: E402
import gen_core  # noqa: E402

DEFAULT_WORKSPACE = Path(r"C:\IA\Breeze-tts\voices")
SR = 24_000


# ------------------------------------------------------------------ texto
def split_blocks(text: str, max_words: int, min_words: int = 5) -> list[str]:
    """Quebra o texto em blocos de <= max_words palavras, cortando em pontuacao forte.

    Tolerancia: uma frase ate ~15 % acima do orcamento fica inteira (evita cortes
    artificiais no meio de uma frase por 1-2 palavras). Blocos muito curtos
    (< min_words) sao grudados no vizinho -- blocos de 1-2 palavras geram audio de
    ~1,5 s, que e ruim para estabilidade e para a medicao de identidade.
    """
    sents = [s.strip() for s in re.split(r"(?<=[.!?;:])\s+", text.strip()) if s.strip()]
    soft = max(max_words, int(round(max_words * 1.15)))
    blocks: list[str] = []
    cur: list[str] = []
    n = 0
    for s in sents:
        w = len(s.split())
        if cur and n + w > soft:
            blocks.append(" ".join(cur))
            cur, n = [], 0
        if w > soft:                           # frase gigante: fatia por virgula e, se
            parts = [p.strip() for p in s.split(",") if p.strip()]   # ainda faltar, por palavras
            chunks: list[str] = []
            for pi, p in enumerate(parts):
                suffix = "," if pi < len(parts) - 1 else ""
                pw = p.split()
                if len(pw) <= max_words:
                    chunks.append(p + suffix)
                    continue
                for i in range(0, len(pw), max_words):
                    is_last = i + max_words >= len(pw)
                    chunks.append(" ".join(pw[i:i + max_words]) + (suffix if is_last else ""))
            for ch in chunks:
                cw = len(ch.split())
                if cur and n + cw > soft:
                    blocks.append(" ".join(cur))
                    cur, n = [], 0
                cur.append(ch)
                n += cw
            continue
        cur.append(s)
        n += w
    if cur:
        blocks.append(" ".join(cur))

    # junta blocos curtos no vizinho (curto no inicio -> proximo; no fim -> anterior)
    out: list[str] = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        if len(b.split()) < min_words and i + 1 < len(blocks):
            blocks[i + 1] = b + " " + blocks[i + 1]
            i += 1
            continue
        out.append(b)
        i += 1
    if len(out) > 1 and len(out[-1].split()) < min_words:
        out[-2] = out[-2] + " " + out[-1]
        out.pop()
    return out


# ------------------------------------------------------------------ audio
def dur_ok(dur: float, words: int, lo: float = 0.55, hi: float = 2.2,
           wps: float = 2.6) -> bool:
    """A duracao gerada e plausivel para o texto? (wps ~ 155 palavras/min)

    O gate relativo (dur vs mediana dos candidatos) nao pega o caso de TODOS os
    candidatos serem longos -- e nesse modelo um texto curto pode virar 13 s de fala
    arrastada. Este gate ancora a duracao no TEXTO, nao nos vizinhos.
    """
    exp_s = max(2.0, words / max(wps, 0.1))
    return lo * exp_s <= dur <= hi * exp_s


def rms_match(wav: np.ndarray, target_rms: float, max_gain: float = 4.0) -> np.ndarray:
    cur = float(np.sqrt(np.mean(wav ** 2))) if len(wav) else 0.0
    if cur < 1e-6:
        return wav
    g = float(np.clip(target_rms / cur, 1.0 / max_gain, max_gain))
    return np.clip(wav * g, -1.0, 1.0).astype("float32")


def load_refs(workspace: Path, voice: str, extra: list[str]) -> tuple[list[Path], str, dict]:
    vdir = workspace / voice
    meta = {}
    if (vdir / "voice.json").is_file():
        meta = json.loads((vdir / "voice.json").read_text(encoding="utf-8"))
    refs: list[Path] = []
    if (vdir / "ref.wav").is_file():
        refs.append(vdir / "ref.wav")
    refs += sorted((vdir / "refs").glob("*.wav")) if (vdir / "refs").is_dir() else []
    refs += [Path(p) for p in extra if Path(p).is_file()]
    if not refs:
        raise SystemExit(f"[gen] nenhuma referencia em {vdir} (esperado ref.wav)")
    rtxt = ""
    if (vdir / "ref.txt").is_file():
        rtxt = (vdir / "ref.txt").read_text(encoding="utf-8").strip()
    rtxt = meta.get("ref_text") or rtxt
    if not rtxt:
        raise SystemExit("[gen] ref.txt vazio: transcreva a referencia antes de gerar")
    return refs, rtxt, meta


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description="Geracao consistente com medoid reranking")
    ap.add_argument("--voice", default=None, help="nome da pasta em --workspace")
    ap.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    ap.add_argument("--ref-audio", default=None, help="referencia direta (alternativa a --voice)")
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--extra-refs", default="", help="outras gravacoes do locutor (csv)")
    ap.add_argument("--adapter", default=str(gen_core.DEFAULT_ADAPTER))
    ap.add_argument("--adapter-scale", type=float, default=1.0,
                    help="escala do LoRA na inferencia (a literatura de producao usa 0.3-0.5; "
                         "1.0 pode 'over-steer')")
    ap.add_argument("--text", default=None)
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--instruction", default="Fale com clareza e naturalidade.")
    ap.add_argument("--template", default="ref_clone_tata",
                    choices=["ref_clone_tata", "ref_edit_tata", "tts_instruction"])
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--use-dual-cfg", action="store_true")
    ap.add_argument("--cfg-ref", type=float, default=1.2)
    ap.add_argument("--cfg-ins", type=float, default=1.0)
    ap.add_argument("--max-block-s", type=float, default=10.0,
                    help="teto de duracao por bloco, em segundos (default 10; use 8 p/ mais fidelidade)")
    ap.add_argument("--max-words", type=int, default=None,
                    help="override do teto em palavras (se omitido, derivado de --max-block-s)")
    ap.add_argument("--min-dur-ratio", type=float, default=0.55,
                    help="piso de duracao plausivel (x tempo esperado do texto)")
    ap.add_argument("--max-dur-ratio", type=float, default=2.2,
                    help="teto de duracao plausivel (x tempo esperado do texto)")
    ap.add_argument("--max-wer", type=float, default=0.05, help="gate de inteligibilidade")
    ap.add_argument("--pause-s", type=float, default=0.35, help="pausa entre blocos")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-asr", action="store_true", help="pula WER (mais rapido)")
    a = ap.parse_args()

    text = a.text or (Path(a.text_file).read_text(encoding="utf-8").strip() if a.text_file else None)
    if not text:
        raise SystemExit("[gen] informe --text ou --text-file")

    workspace = Path(a.workspace)
    if a.ref_audio:
        refs = [Path(a.ref_audio)]
        rtxt = (a.ref_text or "").strip()
        if not rtxt:
            raise SystemExit("[gen] --ref-audio exige --ref-text")
        meta = {}
    else:
        if not a.voice:
            raise SystemExit("[gen] informe --voice ou --ref-audio")
        refs, rtxt, meta = load_refs(workspace, a.voice,
                                     [p.strip() for p in a.extra_refs.split(",") if p.strip()])

    WPS = 2.6                                    # ~155 palavras/min (narracao pt-BR)
    max_words = a.max_words or max(6, int(round(a.max_block_s * WPS)))
    blocks = split_blocks(text, max_words)
    print(f"[gen] teto por bloco: {a.max_block_s:.1f}s (~{max_words} palavras, {WPS} palavras/s)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[gen] adapter   : {a.adapter}")
    print(f"[gen] template  : {a.template} (cfg_scale={a.cfg_scale}, dual={a.use_dual_cfg})")
    print(f"[gen] referencias ({len(refs)}): " + ", ".join(p.name for p in refs))
    print(f"[gen] {len(blocks)} bloco(s) x {a.candidates} candidatos "
          f"= {len(blocks)*a.candidates} geracoes (~{len(blocks)*a.candidates*2.3:.0f} min)")

    # template efetivamente disponivel no engine
    from breeze_infer.templates import TEMPLATES

    template = a.template if a.template in TEMPLATES else "ref_edit_tata"
    if template != a.template:
        print(f"[gen] (aviso) '{a.template}' indisponivel neste engine -> usando '{template}' "
              "(atualize o clone do engine para ter o clone puro)")
    if template == "ref_clone_tata" and a.cfg_scale != 1.0:
        print("[gen] (aviso) ref_clone_tata NAO tem ramo negativo no CFG -> forcando "
              "cfg_scale=1.0 (recomendacao oficial do Breeze para Voice Clone)")
        a.cfg_scale = 1.0

    cfg = dict(temperature=a.temperature, top_k=a.top_k, top_p=a.top_p, cfg_scale=a.cfg_scale,
               use_dual_cfg=a.use_dual_cfg, cfg_ref=a.cfg_ref, cfg_ins=a.cfg_ins,
               template=template, instruction=a.instruction, max_new_tokens=400)

    refs_clean: list[Path] = []
    import librosa
    for r in refs:                                   # referencias normalizadas (mono 24k/trim/norm)
        w, sr = sf.read(str(r), dtype="float32", always_2d=True)
        w = w.mean(axis=1)
        if sr != SR:
            w = librosa.resample(w, orig_sr=sr, target_sr=SR)
        w, _ = librosa.effects.trim(w, top_db=40)
        pk = float(np.max(np.abs(w))) if len(w) else 0.0
        if pk < 0.30 or pk > 0.99:
            w = w * (0.95 / max(pk, 1e-9))
        p = out / f"_ref_{r.stem}.wav"
        sf.write(str(p), np.clip(w, -1, 1), SR, subtype="PCM_16")
        refs_clean.append(p)
    ref_emb = [metrics.embed(str(p), "cpu") for p in refs_clean]
    print("[gen] ref_wav normalizada: "
          + ", ".join(f"{p.name}={sf.info(str(p)).duration:.2f}s" for p in refs_clean))

    model, tok, atok = gen_core.load_model(a.adapter, a.device)
    if a.adapter_scale != 1.0:
        n_sc = gen_core.apply_adapter_scale(model, a.adapter_scale)
        print(f"[gen] escala do adapter = {a.adapter_scale} aplicada em {n_sc} modulos LoRA")
    cfg = {**cfg, "adapter_scale": a.adapter_scale}
    rows: list[dict] = []
    chosen_wavs: list[np.ndarray] = []
    manifest: list[dict] = []
    t0 = time.time()

    for bi, btext in enumerate(blocks, 1):
        bdir = out / f"bloco{bi:02d}"
        bdir.mkdir(exist_ok=True)
        (bdir / "texto.txt").write_text(btext + "\n", encoding="utf-8")
        print(f"\n[gen] bloco {bi}/{len(blocks)} ({len(btext.split())} palavras): {btext[:70]}...")
        cand: list[dict] = []
        for k in range(a.candidates):
            seed = a.seed_base + k
            wav_p = bdir / f"cand{k:02d}_seed{seed:03d}.wav"
            if not wav_p.exists():
                wav, sr = gen_core.generate_one(model, tok, atok, cfg, seed,
                                                 refs_clean[0], rtxt, btext, a.device)
                sf.write(str(wav_p), np.clip(wav, -1, 1), sr, subtype="PCM_16")
            e = metrics.embed(str(wav_p), "cpu")
            cos_ref = float(np.mean([metrics.cos(e, re) for re in ref_emb]))
            cf, cl, drift = metrics.temporal_cos(str(wav_p), ref_emb[0], "cpu")
            dur = sf.info(str(wav_p)).frames / SR
            hyp, wer = "", float("nan")
            if not a.no_asr:
                try:
                    hyp = metrics.transcribe(str(wav_p), device="cpu")
                    wer, _ = metrics.wer_cer(btext, hyp)
                except Exception as exc:  # noqa: BLE001
                    print(f"      (aviso) ASR falhou: {type(exc).__name__}")
            r = {"bloco": bi, "seed": seed, "wav": str(wav_p), "cos_ref_multi": round(cos_ref, 4),
                 "cos_ref_1a": round(metrics.cos(e, ref_emb[0]), 4), "drift": round(drift, 4),
                 "dur": round(dur, 2), "wer": round(wer, 4) if wer == wer else "",
                 "hyp": hyp[:120]}
            rows.append(r)
            cand.append({**r, "emb": e})
            print(f"      seed {seed:3d}: cos_multi={cos_ref:.4f} drift={drift:+.4f} "
                  f"dur={dur:5.2f}s wer={wer if wer == wer else float('nan'):.3f}"
                  f" ({time.time()-t0:.0f}s decorridos)")

        # gate -> medoid  (relativo entre candidatos + ancorado no texto)
        med_dur = float(np.median([x["dur"] for x in cand]))
        n_words = len(btext.split())
        ok = [c for c in cand
              if (c["wer"] == "" or c["wer"] <= a.max_wer)
              and 0.75 <= c["dur"] / max(0.01, med_dur) <= 1.30
              and dur_ok(c["dur"], n_words, a.min_dur_ratio, a.max_dur_ratio)]
        rejeitados_dur = [c["seed"] for c in cand
                          if not dur_ok(c["dur"], n_words, a.min_dur_ratio, a.max_dur_ratio)]
        pool = ok or cand
        if not ok:
            print(f"    (aviso) nenhum candidato passou no gate ({len(cand)} testados"
                  + (f"; duracao fora do plausivel: seeds {rejeitados_dur}" if rejeitados_dur else "")
                  + ") — usando todos. Gere mais candidatos ou revise o texto.")
        if len(pool) == 1:
            winner = pool[0]
        else:
            E = np.stack([c["emb"] for c in pool])
            E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
            S = E @ E.T
            np.fill_diagonal(S, np.nan)
            winner = pool[int(np.nanargmax(np.nanmean(S, axis=1)))]
        best_cos = max(c["cos_ref_multi"] for c in pool)
        print(f"    -> escolhido: seed {winner['seed']} (medoid) cos_multi={winner['cos_ref_multi']:.4f} "
              f"| melhor candidato tinha {best_cos:.4f} "
              f"(gate: {len(ok)}/{len(cand)} passaram)")
        for c in cand:
            c.pop("emb", None)
        (bdir / "escolhido.json").write_text(
            json.dumps({"texto": btext, **winner, "n_passaram_gate": len(ok),
                        "melhor_cos_do_lote": best_cos}, indent=1, ensure_ascii=False),
            encoding="utf-8")
        w, _ = sf.read(str(winner["wav"]), dtype="float32")
        chosen_wavs.append(w)
        manifest.append({"bloco": bi, "texto": btext, **winner})

    with (out / "report.csv").open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    tgt = float(np.mean([np.sqrt(np.mean(w ** 2)) for w in chosen_wavs]))
    pause = np.zeros(int(a.pause_s * SR), dtype="float32")
    parts: list[np.ndarray] = []
    for w in chosen_wavs:
        parts.append(rms_match(w.astype("float32"), tgt))
        parts.append(pause)
    final = np.concatenate(parts[:-1]) if parts else np.zeros(0, dtype="float32")
    sf.write(str(out / "final.wav"), final, SR, subtype="PCM_16")
    (out / "manifest.json").write_text(
        json.dumps({"adapter": a.adapter, "template": template, "cfg": cfg,
                    "refs": [str(p) for p in refs_clean],
                    "blocos": manifest, "final": str(out / "final.wav"),
                    "rms_alvo": tgt}, indent=1, ensure_ascii=False), encoding="utf-8")

    med = float(np.median([m["cos_ref_multi"] for m in manifest]))
    dr = float(np.mean([m["drift"] for m in manifest]))
    wers = [m["wer"] for m in manifest if m["wer"] != ""]
    print("\n" + "=" * 78)
    print(f"[gen] FINAL: {out / 'final.wav'} ({len(final)/SR:.1f}s)")
    print(f"[gen] mediana cos_ref_multi por bloco = {med:.4f} | drift medio = {dr:+.4f} | "
          f"WER medio = {np.mean(wers) if wers else float('nan'):.4f}")
    print(f"[gen] relatorio: {out / 'report.csv'} | manifest: {out / 'manifest.json'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
