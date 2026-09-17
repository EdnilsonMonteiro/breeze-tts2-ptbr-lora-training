"""compare_seeds.py — metricas + ranking das geracoes do sweep.

Para cada WAV calcula:
- cos ECAPA vs referencia longa (VozEdnilson.wav) e vs frase exata (FalandoFrase);
- WER/CER (faster-whisper) vs texto-alvo + acerto das palavras-alvo;
- prosodia (duracao, F0, RMS).

Saidas: metrics.csv, ranking.md e plots/ (hist, cos x WER, boxplot por config).

Uso:
  python compare_seeds.py --dir teste_seeds --out teste_seeds/metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import metrics

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description="Metricas/ranking do sweep de seeds.")
    ap.add_argument("--dir", default=str(HERE / "teste_seeds"))
    ap.add_argument("--ref", default=str(HERE / "VozEdnilson.wav"))
    ap.add_argument("--ref-phrase", default=str(HERE / "VozEdnilsonFalandoFrase.wav"))
    ap.add_argument("--text-file", default=str(HERE / "frase_alvo.txt"))
    ap.add_argument("--words", default="carro,arroz,lataria,barroso,fixo")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    root = Path(args.dir)
    wavs = sorted(root.rglob("*.wav"))
    if not wavs:
        sys.exit(f"[cmp] nenhum wav em {root}")
    out_csv = Path(args.out) if args.out else root / "metrics.csv"
    text = Path(args.text_file).read_text(encoding="utf-8").strip()
    words = [w.strip() for w in args.words.split(",") if w.strip()]

    e_ref = metrics.embed(args.ref, args.device)
    ref_phrase = Path(args.ref_phrase)
    e_phrase = metrics.embed(str(ref_phrase), args.device) if ref_phrase.is_file() else None
    print(f"[cmp] ref self-similarity (ref vs frase) = "
          f"{metrics.cos(e_ref, e_phrase):.3f}" if e_phrase is not None else "[cmp] sem frase")

    rows = []
    for i, w in enumerate(wavs, 1):
        e = metrics.embed(str(w), args.device)
        cos_ref = metrics.cos(e_ref, e)
        cos_phrase = metrics.cos(e_phrase, e) if e_phrase is not None else float("nan")
        hyp = metrics.transcribe(str(w), device="cpu")
        wer, cer = metrics.wer_cer(text, hyp)
        hits = metrics.word_hits(text, hyp, words)
        pr = metrics.prosody(str(w))
        id_score = np.nanmean([cos_ref, cos_phrase])
        score = 0.6 * id_score + 0.4 * (1.0 - min(wer, 1.0))
        rows.append({
            "tag": w.parent.name, "seed": w.stem, "wav": str(w),
            "cos_ref": round(cos_ref, 4), "cos_phrase": round(cos_phrase, 4),
            "wer": round(wer, 4), "cer": round(cer, 4), "score": round(float(score), 4),
            **{f"hit_{k}": int(v) for k, v in hits.items()},
            **pr, "hyp": hyp,
        })
        print(f"[cmp] {i}/{len(wavs)} {w.parent.name}/{w.name}: "
              f"cos={cos_ref:.3f} ph={cos_phrase:.3f} wer={wer:.3f} score={score:.3f}",
              flush=True)

    fields = list(rows[0].keys())
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=fields)
        wcsv.writeheader()
        wcsv.writerows(rows)
    print(f"[cmp] -> {out_csv}")

    # -------------------------------------------------- ranking por config (tag)
    by_tag = defaultdict(list)
    for r in rows:
        by_tag[r["tag"]].append(r)

    def agg(rs):
        return {
            "n": len(rs),
            "cos_ref": float(np.mean([r["cos_ref"] for r in rs])),
            "cos_ref_std": float(np.std([r["cos_ref"] for r in rs])),
            "cos_phrase": float(np.nanmean([r["cos_phrase"] for r in rs])),
            "wer": float(np.mean([r["wer"] for r in rs])),
            "score": float(np.mean([r["score"] for r in rs])),
            "score_std": float(np.std([r["score"] for r in rs])),
            "dur": float(np.mean([r["dur"] for r in rs])),
        }

    cfg_rows = sorted(((t, agg(rs)) for t, rs in by_tag.items()),
                      key=lambda x: -x[1]["score"])
    ranking = out_csv.parent / "ranking.md"
    lines = ["# Ranking do sweep de seeds", "",
             f"- texto-alvo: `{text[:70]}...`",
             f"- `score = 0.6*identidade(cos) + 0.4*(1-WER)`", "",
             "## Por configuracao (media)", "",
             "| tag | n | cos_ref | cos_phr | WER | score | dur(s) |",
             "|---|---|---|---|---|---|---|"]
    for t, a in cfg_rows:
        lines.append(f"| `{t}` | {a['n']} | {a['cos_ref']:.3f} | {a['cos_phrase']:.3f} | "
                     f"{a['wer']:.3f} | **{a['score']:.3f}** | {a['dur']:.1f} |")
    top = sorted(rows, key=lambda r: -r["score"])[:10]
    lines += ["", "## Top 10 seeds (global)", "",
              "| seed | tag | cos_ref | WER | score |", "|---|---|---|---|---|"]
    for r in top:
        lines.append(f"| {r['seed']} | `{r['tag']}` | {r['cos_ref']:.3f} | "
                     f"{r['wer']:.3f} | {r['score']:.3f} |")
    ranking.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[cmp] -> {ranking}")
    print("\n".join(lines))

    if not args.no_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            pdir = out_csv.parent / "plots"
            pdir.mkdir(exist_ok=True)
            plt.figure(); plt.hist([r["cos_ref"] for r in rows], bins=15)
            plt.xlabel("cos_ref"); plt.title("Distribuicao cos vs referencia"); plt.tight_layout()
            plt.savefig(pdir / "hist_cos_ref.png", dpi=110); plt.close()

            plt.figure()
            plt.scatter([r["cos_ref"] for r in rows], [r["wer"] for r in rows])
            plt.xlabel("cos_ref"); plt.ylabel("WER"); plt.title("Identidade x Inteligibilidade")
            plt.tight_layout(); plt.savefig(pdir / "cos_vs_wer.png", dpi=110); plt.close()

            plt.figure()
            tags = [t for t, _ in cfg_rows]
            data = [[r["cos_ref"] for r in by_tag[t]] for t in tags]
            plt.boxplot(data)
            plt.xticks(range(1, len(tags) + 1), tags, rotation=20, ha="right")
            plt.ylabel("cos_ref")
            plt.title("cos_ref por config")
            plt.tight_layout(); plt.savefig(pdir / "boxplot_cos_por_config.png", dpi=110); plt.close()
            print(f"[cmp] plots -> {pdir}")
        except Exception as exc:  # noqa: BLE001
            print(f"[cmp] (aviso) plots falharam: {exc}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
