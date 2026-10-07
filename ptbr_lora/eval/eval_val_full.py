"""eval_val_full.py — avalia val/test COMPLETO com o protocolo do treino v3 (cross-ref).

Protocolo: cada item usa OUTRO clipe do mesmo locutor como referencia (condicao fixa,
epoca 0, mix so com variantes de referencia); itens sem referencia possivel sao descartados.
Cada item e um forward (batch 1) e e ponderado pelo n.o de frames supervisionados;
backbone e depth decoder sao reportados separadamente, mais por corpus e IC95 bootstrap.

ATENCAO: o split v3 (por grupo, sem vazamento) NAO e o mesmo das runs antigas — os numeros
antigos de val_loss nao sao comparaveis com estes. Compare adapters entre si SOB O MESMO split;
o base (sem adapter) e sempre avaliado primeiro como referencia.

Uso:
  python eval_val_full.py --adapters <a1> <a2> ... --out <json>
  python eval_val_full.py --split test --adapters <a1> --out <json>
  python eval_val_full.py --limit 64 --adapters <a1> --out <json>     # smoke
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

_PROJ = Path(__file__).resolve().parents[2]
_CORE = _PROJ / "ptbr_lora" / "core"
for _p in (str(_CORE), str(_PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import common_breeze as CB  # noqa: E402
from train_lora import TrainDataset, load_split  # noqa: E402

DEV = "cuda"


def _wmean(pairs: list[tuple[float, float]]) -> float:
    w = sum(p[1] for p in pairs)
    return sum(p[0] * p[1] for p in pairs) / w if w else float("nan")


def _boot_ci(pairs: list[tuple[float, float]], n_boot: int = 1000, seed: int = 0):
    if not pairs:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(pairs)
    ms = sorted(_wmean([pairs[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    return ms[int(0.025 * n_boot)], ms[min(n_boot - 1, int(0.975 * n_boot))]


def eval_loss(raw, ds, limit: int | None) -> dict:
    raw.eval()
    n = len(ds) if limit is None else min(len(ds), limit)
    step = max(1, len(ds) // n) if limit else 1
    tot, bb, dd = [], [], []
    per: dict[str, list] = {}
    with torch.no_grad():
        for k in range(0, len(ds), step):
            if len(tot) >= n:
                break
            batch = CB.collate([ds[k]])
            w = float((batch["labels"] == CB.AUDIO_TOKEN_ID).sum().item()) or 1.0
            batch = {a: (v.to(DEV) if isinstance(v, torch.Tensor) else v) for a, v in batch.items()}
            out = raw(**batch)
            vals = (out.loss.item(), out.backbone_loss.item(), out.depth_decoder_loss.item())
            if not all(math.isfinite(v) for v in vals):
                continue
            tot.append((vals[0], w))
            bb.append((vals[1], w))
            dd.append((vals[2], w))
            per.setdefault(ds.records[ds.idxs[k]].get("corpus", "tata"), []).append((vals[0], w))
    lo, hi = _boot_ci(tot)
    return {"loss": _wmean(tot), "ci95": [lo, hi], "backbone": _wmean(bb), "depth": _wmean(dd),
            "per_corpus": {c: _wmean(v) for c, v in sorted(per.items())}, "n": len(tot)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="*", default=[])
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[load] base...", flush=True)
    raw = CB.load_breeze_model(DEV, attn="eager")
    tokenizer = CB.load_text_tokenizer()
    ds = TrainDataset(load_split(args.split), tokenizer, cross_ref_only=True, seed=2)
    print(f"[data] {args.split}={len(ds)} itens cross-ref ({ds.n_dropped_no_ref} sem ref descartados)"
          f" limit={args.limit}", flush=True)

    results = []
    t = time.time()
    r = eval_loss(raw, ds, args.limit)
    print(f"[base] {args.split}_loss={r['loss']:.4f} (bb {r['backbone']:.3f} / depth {r['depth']:.3f}; "
          f"{r['n']} itens, {time.time() - t:.0f}s)", flush=True)
    results.append({"adapter": None, "label": "base", "split": args.split, **r})

    from peft import PeftModel

    for ad in args.adapters:
        ad_path = Path(ad)
        if not ad_path.is_absolute():
            ad_path = (CB.TRAINING / ad).resolve()
        print(f"[adapter] {ad_path.name} ...", flush=True)
        t = time.time()
        pm = PeftModel.from_pretrained(raw, str(ad_path), is_trainable=False)
        r = eval_loss(pm, ds, args.limit)
        print(f"[{ad_path.name}] {args.split}_loss={r['loss']:.4f} "
              f"(IC95 {r['ci95'][0]:.3f}-{r['ci95'][1]:.3f}; bb {r['backbone']:.3f} / "
              f"depth {r['depth']:.3f}; {r['n']} itens, {time.time() - t:.0f}s)", flush=True)
        results.append({"adapter": str(ad_path), "label": ad_path.name, "split": args.split, **r})
        try:
            raw = pm.unload()
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] unload falhou ({exc}); recarregando base", flush=True)
            pm = None
            gc.collect()
            torch.cuda.empty_cache()
            raw = CB.load_breeze_model(DEV, attn="eager")
        pm = None
        gc.collect()
        torch.cuda.empty_cache()

    results_sorted = sorted(results, key=lambda x: x["loss"])
    out_path.write_text(json.dumps(results_sorted, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n[ranking {args.split}] (menor=melhor; diferencas dentro do IC95 nao sao reais)")
    for x in results_sorted:
        print(f"  {x['loss']:.4f}  {x['label']}")
    print(f"[out] {out_path}")


if __name__ == "__main__":
    main()
