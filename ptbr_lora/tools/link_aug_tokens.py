"""link_aug_tokens.py — reaproveita tokens de referencias degradadas (tokens_aug/) de uma base anterior.

Liga (hardlink; copia se o volume nao permitir) `<src>/tokens_aug/<idx>__a<k>.npz` para `<dst>/tokens_aug/`
para todo idx do split de treino de <dst>. Depois, `augment_refs.py --k N` so gera o que falta.
Uso: python ptbr_lora/tools/link_aug_tokens.py --src <artefatos>/training_v5 --dst <artefatos>/training_v6
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--split", default="train")
    a = ap.parse_args(argv)
    want = set((a.dst / f"splits_{a.split}.txt").read_text(encoding="utf-8").split())
    sdir, ddir = a.src / "tokens_aug", a.dst / "tokens_aug"
    ddir.mkdir(parents=True, exist_ok=True)
    n = skip = 0
    for fn in os.listdir(sdir):
        if not fn.endswith(".npz") or "__a" not in fn or fn.rsplit("__a", 1)[0] not in want:
            continue
        dst = ddir / fn
        if dst.exists():
            skip += 1
            continue
        try:
            os.link(sdir / fn, dst)
        except OSError:
            shutil.copy2(sdir / fn, dst)
        n += 1
    print(f"[aug-link] {n} ligados, {skip} ja existiam -> {ddir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
