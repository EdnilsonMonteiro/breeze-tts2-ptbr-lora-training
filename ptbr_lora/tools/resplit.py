"""resplit.py — recalcula os splits treino/val/teste POR GRUPO, sem torch e sem GPU.

Uso:
  python ptbr_lora/tools/resplit.py            # dry-run: mostra estatisticas + vazamentos do split atual
  python ptbr_lora/tools/resplit.py --write    # faz backup do split atual e grava o novo

O `--write` copia os `splits_*.txt` atuais para `<training>/splits_backup_<data>/` antes de
sobrescrever; para reverter, copie-os de volta. ATENCAO: val/test mudam -> val_loss de runs
antigas NAO e comparavel com runs novas (o val antigo tinha vazamento de programa/locutor).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

CORE = Path(__file__).resolve().parents[1] / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

import meta_io  # noqa: E402
import paths  # noqa: E402
import splits as SPL  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="grava (com backup); default = dry-run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fracs", default="0.90,0.05,0.05")
    args = ap.parse_args()
    fracs = tuple(float(x) for x in args.fracs.split(","))

    recs = meta_io.load_meta_records()
    print(f"[resplit] registros={len(recs)}  training={paths.TRAINING}")
    new = SPL.make_splits(recs, fracs, args.seed)
    audit = SPL.audit_leaks(recs, new)
    assert not audit["group_leaks"] and not audit["speaker_leaks"], audit
    stats = SPL.split_stats(recs, new)

    old = {n: SPL.load_split_file(paths.TRAINING, n) for n in SPL.SPLITS}
    if any(old.values()):
        oa = SPL.audit_leaks(recs, old)
        print(f"[resplit] split ATUAL: { {k: len(v) for k, v in old.items()} }")
        print(f"[resplit]   grupos vazando entre splits: {len(oa['group_leaks'])}  "
              f"(ex.: {oa['group_leaks'][:3]})  locutores: {len(oa['speaker_leaks'])}")
    print(f"[resplit] split NOVO : { {k: len(v) for k, v in new.items()} }  vazamentos=0")
    for name, st in stats.items():
        print(f"  {name}: {json.dumps(st, ensure_ascii=False)}")

    if not args.write:
        print("[resplit] dry-run: nada gravado (use --write).")
        return
    if any(old.values()):
        bk = paths.TRAINING / f"splits_backup_{dt.datetime.now():%Y%m%d_%H%M%S}"
        bk.mkdir(parents=True)
        for n in SPL.SPLITS:
            src = paths.TRAINING / f"splits_{n}.txt"
            if src.exists():
                shutil.copy2(src, bk / src.name)
        print(f"[resplit] backup do split atual em {bk}")
    SPL.write_splits(new, paths.TRAINING)
    (paths.TRAINING / "splits_meta.json").write_text(json.dumps(
        {"seed": args.seed, "fracs": fracs, "group_rule": "tagarela/podcast: speaker sem sufixo ':N'",
         "stats": stats, "audit": audit}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[resplit] splits gravados.")


if __name__ == "__main__":
    main()
