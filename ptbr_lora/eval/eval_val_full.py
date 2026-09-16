"""eval_val_full.py — avalia o val COMPLETO (1287 itens) com a MESMA loss do
quick_val_loss (CB.collate + raw(**batch)), para ranking confiavel de adapters.
O mini-val de 24 itens usado no treino tem ruido > 0.004 (as deltas das ultimas
rodadas) — este script resolve isso.

Uso:
  python eval_val_full.py --adapters <a1> <a2> ... --batch 4 --out <json>
  python eval_val_full.py --limit 64 --adapters <a1>          # smoke
Base (sem adapter) e sempre avaliado primeiro como referencia.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PROJ = Path(__file__).resolve().parents[2]
_CORE = _PROJ / "ptbr_lora" / "core"
for _p in (str(_CORE), str(_PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import common_breeze as CB  # noqa: E402
from train_lora import TrainDataset, load_split  # noqa: E402

DEV = "cuda"


def eval_loss(raw, ds, batch_size: int, limit: int | None) -> tuple[float, int]:
    raw.eval()
    n = len(ds) if limit is None else min(len(ds), limit)
    losses = []
    with torch.no_grad():
        for j in range(0, n, batch_size):
            idxs = list(range(j, min(j + batch_size, n)))
            items = [ds[k] for k in idxs]
            batch = CB.collate(items)
            batch = {
                k: (v.to(DEV) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            out = raw(**batch)
            losses.append(out.loss.item())
    return float(np.mean(losses)), n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="*", default=[])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[load] base...", flush=True)
    raw = CB.load_breeze_model(DEV, attn="eager")
    tokenizer = CB.load_text_tokenizer()
    ds_val = TrainDataset(load_split("val"), tokenizer)
    print(f"[data] val={len(ds_val)} itens (limit={args.limit})", flush=True)

    results = []

    t = time.time()
    loss, n = eval_loss(raw, ds_val, args.batch, args.limit)
    print(f"[base] val_full={loss:.4f} ({n} itens, {time.time()-t:.0f}s)", flush=True)
    results.append({"adapter": None, "label": "base", "val_full": loss, "n": n})

    from peft import PeftModel

    for ad in args.adapters:
        ad_path = Path(ad)
        if not ad_path.is_absolute():
            ad_path = (CB.TRAINING / ad).resolve()
        print(f"[adapter] {ad_path.name} ...", flush=True)
        t = time.time()
        pm = PeftModel.from_pretrained(raw, str(ad_path), is_trainable=False)
        loss, n = eval_loss(pm, ds_val, args.batch, args.limit)
        print(f"[{ad_path.name}] val_full={loss:.4f} ({n} itens, {time.time()-t:.0f}s)",
              flush=True)
        results.append({"adapter": str(ad_path), "label": ad_path.name,
                        "val_full": loss, "n": n})
        try:
            raw = pm.unload()
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] unload falhou ({exc}); recarregando base", flush=True)
            del pm
            gc.collect()
            torch.cuda.empty_cache()
            raw = CB.load_breeze_model(DEV, attn="eager")
        del pm
        gc.collect()
        torch.cuda.empty_cache()

    results_sorted = sorted(results, key=lambda r: r["val_full"])
    out_path.write_text(json.dumps(results_sorted, indent=1), encoding="utf-8")
    print("\n[ranking val_full] (menor=melhor)")
    for r in results_sorted:
        print(f"  {r['val_full']:.4f}  {r['label']}")
    print(f"[out] {out_path}")


if __name__ == "__main__":
    main()
