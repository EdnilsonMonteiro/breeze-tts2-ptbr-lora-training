"""eval_zero_shot.py — avaliacao objetiva ZERO-SHOT no split de TESTE (locutores nao vistos).

Protocolo (o que faltava: as metricas antigas mediam so a voz de referencia usada no treino/UI):
  * itens do TEST, estratificados por corpus (cota ~ raiz do tamanho) e por grupo;
  * prompt = OUTRO clipe do mesmo locutor (refs.pick_ref sobre o pool do test);
  * gera com a MESMA condicao do uso real (ref_edit_tata, instrucao de producao, texto normalizado);
  * WER/CER (Whisper large-v3, normalizado nos dois lados, sem VAD) + S/D/I + falhas catastroficas;
  * SECS (ECAPA) contra clipes HELD-OUT do locutor (nunca o prompt => sem vies de canal), com
    - teto  = cos(clipe real alvo, held-out do locutor)      (o melhor possivel)
    - piso  = cos(geracao, held-out de OUTRO locutor)         (controle negativo)
    - SECS normalizado = (gen - piso) / (teto - piso)
  * IC95 bootstrap e taxa de falha por corpus e no agregado (macro por corpus).

Uso:
  python eval_zero_shot.py --adapter <pasta|''> --n 200 --out <dir>       # '' = modelo base
  python eval_zero_shot.py --adapter A --out X --phase eval                 # so recalcula metricas
O gerador e retomavel (pula wavs existentes). GPU so na fase de geracao.
NAO foi executado no ambiente de desenvolvimento (sem GPU): rode `--n 8` como smoke primeiro.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

_PROJ = Path(__file__).resolve().parents[2]
_CORE = _PROJ / "ptbr_lora" / "core"
_TOOLS = _PROJ / "ptbr_lora" / "tools"
for _p in (str(_CORE), str(_TOOLS), str(_PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import meta_io  # noqa: E402
import paths  # noqa: E402
import refs as R  # noqa: E402
import splits as SPL  # noqa: E402
import text_norm  # noqa: E402
import adapter_scale as AS  # noqa: E402
import asr_metrics as AM  # noqa: E402


def select_items(recs: list[dict], split_idxs: list[str], n: int, seed: int = 0,
                 min_per_corpus: int = 8) -> list[tuple[dict, dict]]:
    """[(rec_alvo, rec_ref)] estratificado por corpus, espalhado por grupo, deterministico."""
    keep = set(split_idxs)
    pool_recs = [r for r in recs if r["idx"] in keep]
    pools = R.build_pools(recs, split_idxs)
    by_idx = {r["idx"]: r for r in recs}
    by_corp: dict[str, list[dict]] = {}
    for r in pool_recs:
        if R.pick_ref(r, pools, 0, seed) is not None and 3.0 <= float(r.get("dur_proc_s", 0)) <= 10.2:
            by_corp.setdefault(r.get("corpus", "tata"), []).append(r)
    if not by_corp:
        return []
    root = {c: math.sqrt(len(v)) for c, v in by_corp.items()}
    tot = sum(root.values())
    out: list[tuple[dict, dict]] = []
    for c in sorted(by_corp):
        lst = sorted(by_corp[c], key=lambda r: (SPL.group_key(r), r["idx"]))
        k = min(len(lst), max(min_per_corpus, round(n * root[c] / tot)))
        step = len(lst) / k
        for j in range(k):
            rec = lst[int(j * step)]
            out.append((rec, by_idx[R.pick_ref(rec, pools, 0, seed)]))
    return out


def heldout_refs(rec: dict, prompt: dict, recs_by_spk: dict[str, list[dict]], k: int = 3) -> list[dict]:
    """Ate k clipes do MESMO locutor que nao sao o alvo nem o prompt (para o SECS sem vies)."""
    cands = [r for r in recs_by_spk.get(rec.get("speaker"), [])
             if r["idx"] not in (rec["idx"], prompt["idx"])]
    cands.sort(key=lambda r: r["idx"])
    if len(cands) <= k:
        return cands
    step = len(cands) / k
    return [cands[int(i * step)] for i in range(k)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="pasta do adapter; '' = modelo base")
    ap.add_argument("--adapter-scale", type=float, default=1.0)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", choices=["all", "gen", "eval"], default="all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--asr-size", default="large-v3")
    ap.add_argument("--asr-device", default="cuda")
    ap.add_argument("--asr-compute", default="float16")
    ap.add_argument("--embed-device", default="cpu")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--template", default="ref_edit_tata")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    recs = meta_io.load_meta_records()
    split_idxs = SPL.load_split_file(paths.TRAINING, args.split)
    assert split_idxs, f"split '{args.split}' vazio: rode ptbr_lora/tools/resplit.py --write"
    items = select_items(recs, split_idxs, args.n, args.seed)
    print(f"[zs] {len(items)} itens ({args.split}) | corpora: "
          f"{ {c: sum(1 for r, _ in items if r.get('corpus') == c) for c in sorted({r.get('corpus') for r, _ in items}) } }")
    wav_dir = paths.WAVS24_DIR

    # ------------------------------------------------------------- fase A: geracao
    if args.phase in ("all", "gen"):
        import numpy as np
        import soundfile as sf

        import gen_core

        adapter = args.adapter if args.adapter is not None else gen_core.DEFAULT_ADAPTER
        print(f"[zs] adapter={adapter or '(base)'} scale={args.adapter_scale}")
        model, tok, atok = gen_core.load_model(adapter, args.device)
        if args.adapter_scale != 1.0:
            AS.apply_adapter_scale(model, args.adapter_scale)
        cfg = {"template": args.template, "instruction": "Fale com clareza e naturalidade.",
               "cfg_scale": args.cfg_scale, "use_dual_cfg": False, "cfg_ref": 1.0, "cfg_ins": 1.0,
               "temperature": args.temperature, "top_k": args.top_k, "top_p": 1.0,
               "max_new_tokens": 400, "speaker": "S0"}
        t0 = time.time()
        for k, (rec, ref) in enumerate(items):
            p = out / "wavs" / f"{rec['idx']}.wav"
            if p.exists():
                continue
            try:
                wav, sr = gen_core.generate_one(
                    model, tok, atok, cfg, args.seed * 1000 + k, str(wav_dir / f"{ref['idx']}.wav"),
                    text_norm.normalize(ref["text"]), text_norm.normalize(rec["text"]), args.device)
                sf.write(str(p), np.clip(wav, -1.0, 1.0), int(sr), subtype="PCM_16")
            except Exception as exc:  # noqa: BLE001
                print(f"[zs] ERRO {rec['idx']}: {type(exc).__name__}: {str(exc)[:100]}")
            if (k + 1) % 10 == 0:
                print(f"[zs] gerados {k + 1}/{len(items)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        del model, tok, atok
        try:
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    if args.phase == "gen":
        return

    # ------------------------------------------------------------- fase B: metricas
    import numpy as np
    import soundfile as sf

    import metrics
    from eval_wer import _get_model, transcribe

    asr = _get_model(args.asr_size, args.asr_device, args.asr_compute)
    recs_by_spk: dict[str, list[dict]] = {}
    keep = set(split_idxs)
    for r in recs:
        if r["idx"] in keep and R.eligible_ref_speaker(r.get("speaker")):
            recs_by_spk.setdefault(r["speaker"], []).append(r)
    speakers = sorted(recs_by_spk)
    emb_cache: dict[str, np.ndarray] = {}

    def emb(path: Path) -> np.ndarray:
        key = str(path)
        if key not in emb_cache:
            emb_cache[key] = metrics.embed(path, args.embed_device)
        return emb_cache[key]

    rows: list[dict] = []
    for k, (rec, ref) in enumerate(items):
        p = out / "wavs" / f"{rec['idx']}.wav"
        if not p.exists():
            rows.append({"idx": rec["idx"], "corpus": rec.get("corpus"), "missing": True})
            continue
        text_t = rec["text"]
        hyp = transcribe(asr, p, "pt")
        d = AM.wer_detail(text_t, hyp, "pt", True)
        info = sf.info(str(p))
        dur = info.frames / float(info.samplerate)
        n_words = len(AM.normalize_for_wer(text_t, "pt", True).split())
        why = AM.catastrophic(d["wer"], dur, n_words, hyp)
        held = heldout_refs(rec, ref, recs_by_spk)
        eg = emb(p)
        secs_prompt = metrics.cos(eg, emb(wav_dir / f"{ref['idx']}.wav"))
        secs_held = float(np.mean([metrics.cos(eg, emb(wav_dir / f"{h['idx']}.wav")) for h in held])) \
            if held else float("nan")
        ceil = float(np.mean([metrics.cos(emb(wav_dir / f"{rec['idx']}.wav"),
                                          emb(wav_dir / f"{h['idx']}.wav")) for h in held])) \
            if held else float("nan")
        other_spk = next((s for s in speakers[(speakers.index(rec["speaker"]) + 1) % len(speakers):]
                          if s != rec["speaker"]), None) if rec["speaker"] in speakers else None
        floor = float("nan")
        if other_spk:
            oh = recs_by_spk[other_spk][:3]
            floor = float(np.mean([metrics.cos(eg, emb(wav_dir / f"{h['idx']}.wav")) for h in oh]))
        rows.append({
            "idx": rec["idx"], "corpus": rec.get("corpus"), "speaker": rec.get("speaker"),
            "ref_idx": ref["idx"], "wer": round(d["wer"], 4), "cer": round(d["cer"], 4),
            "S": d["S"], "D": d["D"], "I": d["I"], "n_ref": d["n_ref"], "dur_s": round(dur, 2),
            "dur_gt_s": rec.get("dur_proc_s"), "fail": "|".join(why),
            "secs_heldout": round(secs_held, 4), "secs_prompt": round(secs_prompt, 4),
            "secs_ceiling": round(ceil, 4), "secs_floor": round(floor, 4),
            "n_heldout": len(held), "hyp": hyp[:200],
        })
        if (k + 1) % 20 == 0:
            print(f"[zs] avaliados {k + 1}/{len(items)}", flush=True)

    ok = [r for r in rows if not r.get("missing")]
    with (out / "items.csv").open("w", encoding="utf-8", newline="") as fh:
        if ok:
            wr = csv.DictWriter(fh, fieldnames=list(ok[0].keys()))
            wr.writeheader()
            wr.writerows(ok)

    def agg(rs: list[dict]) -> dict:
        w = AM.summarize([r["wer"] for r in rs], [bool(r["fail"]) for r in rs])
        sh = [r["secs_heldout"] for r in rs if r["secs_heldout"] == r["secs_heldout"]]
        norm = [(r["secs_heldout"] - r["secs_floor"]) / max(1e-6, r["secs_ceiling"] - r["secs_floor"])
                for r in rs if all(r[k] == r[k] for k in ("secs_heldout", "secs_floor", "secs_ceiling"))]
        return {
            "n": len(rs), "wer": w, "cer_mean": sum(r["cer"] for r in rs) / max(1, len(rs)),
            "fail_rate": w.get("fail_rate"),
            "secs_heldout": AM.summarize(sh) if sh else None,
            "secs_prompt_mean": sum(r["secs_prompt"] for r in rs) / max(1, len(rs)),
            "secs_ceiling_mean": AM.summarize([r["secs_ceiling"] for r in rs if r["secs_ceiling"] == r["secs_ceiling"]]) or None,
            "secs_floor_mean": AM.summarize([r["secs_floor"] for r in rs if r["secs_floor"] == r["secs_floor"]]) or None,
            "secs_normalized": AM.summarize(norm) if norm else None,
        }

    by_corp = {c: agg([r for r in ok if r["corpus"] == c]) for c in sorted({r["corpus"] for r in ok})}
    summary = {"adapter": args.adapter, "adapter_scale": args.adapter_scale, "split": args.split,
               "n_items": len(items), "n_missing": len(rows) - len(ok), "overall_micro": agg(ok) if ok else None,
               "per_corpus": by_corp,
               "macro_wer": sum(v["wer"]["mean"] for v in by_corp.values()) / max(1, len(by_corp)),
               "macro_fail_rate": sum(v["fail_rate"] for v in by_corp.values()) / max(1, len(by_corp)),
               "settings": {k: v for k, v in vars(args).items()}}
    (out / "results.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str),
                                      encoding="utf-8")
    print(f"\n[zs] macro WER={summary['macro_wer']:.4f} | falhas macro={summary['macro_fail_rate']:.3f}")
    for c, v in by_corp.items():
        sh = v["secs_heldout"]["mean"] if v["secs_heldout"] else float("nan")
        sn = v["secs_normalized"]["mean"] if v["secs_normalized"] else float("nan")
        print(f"  {c:9s} n={v['n']:3d} WER={v['wer']['mean']:.3f} "
              f"(IC {v['wer']['ci95'][0]:.3f}-{v['wer']['ci95'][1]:.3f}) falhas={v['fail_rate']:.2f} "
              f"SECS_heldout={sh:.3f} normalizado={sn:.2f}")
    print(f"[zs] saidas: {out / 'results.json'} | {out / 'items.csv'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
