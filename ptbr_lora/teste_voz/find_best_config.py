"""find_best_config.py — procura a melhor config (temperature/cfg/...) PARA UMA VOZ.

Gera uma grade de configs em sementes fixas, mede (compare_seeds) e escolhe a
melhor por `score = 0.6*identidade(cos) + 0.4*(1-WER)`. Salva `best_config.json`.

Fases:
  1) --build-grid : escreve grid.json (nao gera nada)
  2) (default)    : roda seed_sweep na grade + compare_seeds + escolhe melhor
  3) --analyze    : so le metrics.csv e escolhe a melhor (sem gerar)

Uso:
  python find_best_config.py --seeds 1-5 --out-dir teste_seeds
  python find_best_config.py --analyze teste_seeds/metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def build_grid(temps: list[float], profiles: list[str], top_k: int,
               top_p: float) -> list[dict]:
    prof_defs = {
        "plain": {"cfg_scale": 1.0},
        "cfg2": {"cfg_scale": 2.0},
        "dual12": {"use_dual_cfg": True, "cfg_scale": 1.0, "cfg_ref": 1.2, "cfg_ins": 1.0},
        "dual15": {"use_dual_cfg": True, "cfg_scale": 1.0, "cfg_ref": 1.5, "cfg_ins": 1.0},
        "dual2": {"use_dual_cfg": True, "cfg_scale": 1.0, "cfg_ref": 2.0, "cfg_ins": 1.0},
        "dual3": {"use_dual_cfg": True, "cfg_scale": 1.0, "cfg_ref": 3.0, "cfg_ins": 1.0},
        "dual5": {"use_dual_cfg": True, "cfg_scale": 1.0, "cfg_ref": 5.0, "cfg_ins": 1.0},
    }
    grid = []
    for t in temps:
        for p in profiles:
            grid.append({"temperature": t, "top_k": top_k, "top_p": top_p, **prof_defs[p]})
    return grid


def analyze(metrics_csv: Path, max_wer: float = 0.03) -> dict:
    """Escolhe a melhor config SEM sacrificar pronuncia.

    Criterio: entre as configs com WER medio <= `max_wer` (elegiveis), escolhe a
    de maior identidade media ((cos_ref+cos_phrase)/2). Se nenhuma passar, avisa e
    cai para o ranking por `score` (identidade + 1-WER). Assim o dual-CFG so ganha
    se a pronuncia se mantiver.
    """
    import re

    rows = list(csv.DictReader(metrics_csv.open(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"[cfg] metrics vazio: {metrics_csv}")
    by_tag = defaultdict(list)
    for r in rows:
        by_tag[r["tag"]].append(r)

    def agg(rs):
        n = len(rs)
        cos_ref = sum(float(x["cos_ref"]) for x in rs) / n
        cos_phrase = sum(float(x["cos_phrase"]) for x in rs) / n
        return {
            "n": n,
            "cos_ref": cos_ref,
            "cos_phrase": cos_phrase,
            "identity": (cos_ref + cos_phrase) / 2,
            "wer": sum(float(x["wer"]) for x in rs) / n,
            "score": sum(float(x["score"]) for x in rs) / n,
            "score_std": (sum((float(x["score"]) -
                               sum(float(y["score"]) for y in rs) / n) ** 2 for x in rs) / n) ** 0.5,
        }

    ranked = sorted(((t, agg(rs)) for t, rs in by_tag.items()), key=lambda x: -x[1]["score"])
    eligible = [(t, a) for t, a in ranked if a["wer"] <= max_wer]
    if not eligible:
        print(f"[cfg] (aviso) nenhuma config com WER <= {max_wer}; usando score")
        eligible = ranked
    best_tag, best = max(eligible, key=lambda x: x[1]["identity"])

    best_rows = by_tag[best_tag]
    best_seed_raw = max(best_rows, key=lambda r: float(r["score"]))["seed"]
    m = re.search(r"\d+", str(best_seed_raw))
    best_seed = int(m.group()) if m else best_seed_raw
    result = {"best_tag": best_tag, "best": best, "best_seed": best_seed,
              "max_wer": max_wer, "eligible": [t for t, _ in eligible],
              "ranking": [{"tag": t, **a} for t, a in ranked]}
    return result


def attach_config(res: dict, out: Path) -> dict:
    """Anexa em res['best_config'] os parametros da config vencedora (via grid.json)."""
    grid_p = out / "grid.json"
    if grid_p.is_file():
        try:
            import seed_sweep

            for c in json.loads(grid_p.read_text(encoding="utf-8")):
                if seed_sweep.tag_for(c) == res.get("best_tag"):
                    res["best_config"] = c
                    break
        except Exception as exc:  # noqa: BLE001
            print(f"[cfg] (aviso) nao anexei best_config: {exc}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", default=None, help="metrics.csv: so analisa (nao gera)")
    ap.add_argument("--build-grid", action="store_true")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--seeds", default="1-5")
    ap.add_argument("--temps", default="0.7,0.9")
    ap.add_argument("--profiles", default="plain,dual3")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=1200)
    ap.add_argument("--max-wer", type=float, default=0.03,
                    help="WER medio maximo para uma config ser elegivel (nao sacrificar pronuncia)")
    ap.add_argument("--out-dir", default=str(HERE / "teste_seeds"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out_dir)
    grid = build_grid([float(x) for x in args.temps.split(",")],
                      [p.strip() for p in args.profiles.split(",")],
                      args.top_k, args.top_p)
    grid_p = out / "grid.json"
    out.mkdir(parents=True, exist_ok=True)
    grid_p.write_text(json.dumps(grid, indent=1), encoding="utf-8")

    if args.build_grid:
        print(f"[cfg] grid com {len(grid)} configs -> {grid_p}")
        return

    if args.analyze:
        res = analyze(Path(args.analyze), max_wer=args.max_wer)
        res = attach_config(res, out)
        (out / "best_config.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                              encoding="utf-8")
        print(json.dumps(res, indent=1, ensure_ascii=False))
        return

    print(f"[cfg] gerando grade ({len(grid)} configs) em seeds {args.seeds} ...")
    cmd = [PY, str(HERE / "seed_sweep.py"), "--seeds", args.seeds,
           "--configs", str(grid_p), "--out-dir", str(out),
           "--device", args.device, "--max-new-tokens", str(args.max_new_tokens)]
    if args.adapter:
        cmd += ["--adapter", args.adapter]
    subprocess.run(cmd, check=True)

    print("[cfg] medindo ...")
    subprocess.run([PY, str(HERE / "compare_seeds.py"), "--dir", str(out),
                    "--out", str(out / "metrics.csv")], check=True)

    res = analyze(out / "metrics.csv", max_wer=args.max_wer)
    res = attach_config(res, out)
    (out / "best_config.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                          encoding="utf-8")
    print("\n[cfg] MELHOR CONFIG:", res["best_tag"])
    print(f"[cfg] -> {out / 'best_config.json'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
