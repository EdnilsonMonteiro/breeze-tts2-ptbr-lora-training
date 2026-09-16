"""auto_train.py — orquestrador de runs encadeadas com schedule fresco.

Laco: lanca train_lora.py (2 epocas, lr decaindo 50% por rodada), espera terminar,
le o melhor val da rodada e so continua se melhorou >= --epsilon. Para sozinho em:
  - platô de val (sem melhora >= epsilon)
  - --max-rounds atingido
  - arquivo de parada: crie  training\\AUTO_STOP  (o round em curso TERMINA antes)
  - exit code != 0 do treino (crash) -> aborta

Uso (depois do run base terminar):
  python auto_train.py --resume "training\\runs\\myrun\\checkpoints\\epoch1_valX_XXX"
Opcoes: --max-rounds 6  --epochs 2  --lr 1e-4  --decay 0.5  --epsilon 0.002
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

CORE = Path(__file__).resolve().parents[1] / "core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))
import paths  # noqa: E402

ROOT = paths.REPO
PY = paths.BREEZE_PY
TRAIN_PY = CORE / "train_lora.py"
RUNS = paths.TRAINING / "runs"
STOP_FILE = paths.TRAINING / "AUTO_STOP"

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def ckpt_val(name: str) -> float:
    m = re.search(r"val(\d+)_(\d+)", name)
    return float(f"{m.group(1)}.{m.group(2)}") if m else float("inf")


def best_epoch_ckpt(run: str) -> tuple[float, Path]:
    d = RUNS / run / "checkpoints"
    cands = [(ckpt_val(p.name), p) for p in d.iterdir() if p.is_dir() and "val" in p.name]
    return min(cands) if cands else (float("inf"), None)


def fresh_run_name(base: str) -> str:
    name, k = base, 1
    while (RUNS / name).exists():
        k += 1
        name = f"{base}_{k}"
    return name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", required=True, help="adapter inicial (pasta checkpoint)")
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--decay", type=float, default=0.5)
    ap.add_argument("--epsilon", type=float, default=0.002)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--targets", choices=["attn", "all"], default="attn")
    ap.add_argument("--use-rslora", action="store_true")
    ap.add_argument("--ref-edit-frac", type=float, default=0.9)
    ap.add_argument("--val-items", type=int, default=96)
    ap.add_argument("--corpus-weights", type=str, default=None)
    ap.add_argument("--no-wer", action="store_true")
    args = ap.parse_args()

    best_val = ckpt_val(Path(args.resume).name)
    best_ckpt = Path(args.resume).resolve()
    if not best_ckpt.exists():
        sys.exit(f"[auto] checkpoint nao encontrado: {best_ckpt}")
    lr = args.lr
    hist = []

    for rnd in range(1, args.max_rounds + 1):
        if STOP_FILE.exists():
            print(f"[auto] {STOP_FILE.name} presente -> parando antes da rodada {rnd}")
            break
        run = fresh_run_name(f"auto{rnd:02d}")
        print(f"\n[auto] ===== RODADA {rnd}/{args.max_rounds} run={run} lr={lr:.1e} "
              f"resume={best_ckpt.name} =====", flush=True)
        out = RUNS / f"{run}_console.log"
        err = RUNS / f"{run}_err.log"
        cmd = [str(PY), str(TRAIN_PY), "--run", run, "--epochs", str(args.epochs),
               "--lr", f"{lr:g}", "--rank", str(args.rank), "--alpha", str(args.alpha),
               "--targets", args.targets, "--val-items", str(args.val_items),
               "--ref-edit-frac", f"{args.ref_edit_frac:g}",
               "--resume-adapter", str(best_ckpt)]
        if args.use_rslora:
            cmd.append("--use-rslora")
        if args.corpus_weights:
            cmd += ["--corpus-weights", args.corpus_weights]
        if args.no_wer:
            cmd.append("--no-wer")
        print(f"[auto] cmd: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=open(out, "w", encoding="utf-8"),
            stderr=open(err, "w", encoding="utf-8"))
        while proc.poll() is None:
            time.sleep(60)
        if proc.returncode != 0:
            print(f"[auto] treino saiu com codigo {proc.returncode} -> abortando")
            break

        val, ck = best_epoch_ckpt(run)
        (RUNS / run / "auto_meta.json").write_text(json.dumps(
            {"round": rnd, "lr": lr, "resume": str(best_ckpt), "best_val": val,
             "best_ckpt": str(ck) if ck else None}, indent=1), encoding="utf-8")
        hist.append({"run": run, "lr": lr, "best_val": val})
        print(f"[auto] rodada {rnd}: melhor val={val:.4f} ({ck.name if ck else '?'})", flush=True)

        if val < best_val - args.epsilon:
            best_val, best_ckpt = val, ck
            lr *= args.decay
        else:
            print(f"[auto] platô/overfit (melhora < {args.epsilon}) -> FIM "
                  f"(melhor geral: {best_ckpt} val={best_val:.4f})")
            break
    else:
        print(f"[auto] max-rounds atingido (melhor geral: {best_ckpt} val={best_val:.4f})")

    print("\n[auto] historico:", json.dumps(hist, indent=1))
    print(f"[auto] MELHOR ADAPTER: {best_ckpt}")
    print("[auto] para parar as proximas rodadas:  New-Item training\\AUTO_STOP")


if __name__ == "__main__":
    main()
