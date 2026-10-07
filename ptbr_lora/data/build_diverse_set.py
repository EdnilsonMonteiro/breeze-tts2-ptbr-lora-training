"""build_diverse_set.py — monta a base DIVERSA em locutores (pasta nova, ex.: training_v4).

Parte do `training/` ja processado (tokens .npz + wavs 24 kHz), escolhe so vozes com clipes
suficientes, corta cada voz em N clipes, cria splits por GRUPO com muitas vozes nunca vistas em
val/test e grava tudo numa pasta SEPARADA (nao altera o `training/` original). Tokens e wavs sao
ligados por hardlink (sem duplicar disco); com --link none so grava os metadados.

Saida (out/):
  dataset_meta.jsonl  manifest.csv  splits_{train,val,test}.txt  splits_meta.json
  speaker_table.csv   selection_report.md  [tokens/  wavs24/ (links)]

Uso (Windows):
  set PTBR_ARTIFACTS=/path/to/artifacts
  python ptbr_lora\\data\\build_diverse_set.py --link hardlink
Depois treine com  BREEZE_TRAINING_DIR=<out>.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_CORE = Path(__file__).resolve().parents[1] / "core"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))
import meta_io  # noqa: E402
import paths  # noqa: E402
import splits as SPL  # noqa: E402
import voices as V  # noqa: E402

MANIFEST_COLS = ["idx", "corpus", "speaker", "wav_rel", "text", "dur_src_s", "dur_proc_s", "frames",
                 "variant"]


def link_assets(idxs: list[str], src, out: Path, mode: str) -> dict:
    """`src`: pasta de origem OU lista de pastas (a primeira que tiver o arquivo vence)."""
    srcs = [Path(x) for x in (src if isinstance(src, (list, tuple)) else [src])]
    stats = Counter()
    missing: list[str] = []
    for sub, ext in (("tokens", ".npz"), ("wavs24", ".wav")):
        (out / sub).mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for k, idx in enumerate(idxs):
            d = out / sub / f"{idx}{ext}"
            s = next((x / sub / f"{idx}{ext}" for x in srcs if (x / sub / f"{idx}{ext}").exists()),
                     srcs[0] / sub / f"{idx}{ext}")
            if d.exists():
                stats[f"{sub}_ja_existia"] += 1
                continue
            if not s.exists():
                missing.append(f"{sub}/{idx}{ext}")
                continue
            if mode == "hardlink":
                try:
                    os.link(s, d)
                    stats[f"{sub}_link"] += 1
                    continue
                except OSError:
                    stats["fallback_copy"] += 1
            shutil.copy2(s, d)
            stats[f"{sub}_copy"] += 1
            if (k + 1) % 5000 == 0:
                print(f"[link] {sub} {k + 1}/{len(idxs)} ({time.time() - t0:.0f}s)", flush=True)
    return {"stats": dict(stats), "missing": missing}


def write_report(path: Path, cfg: V.VoiceCfg, recs_all, sel, splits, info, summ, audit) -> None:
    by_c_all = Counter(r.get("corpus", "tata") for r in recs_all)
    by_c_sel = Counter(r.get("corpus", "tata") for r in sel)
    vox_all = defaultdict(set)
    for r in recs_all:
        vox_all[r.get("corpus", "tata")].add(V.voice_key(r))
    vox_sel = defaultdict(set)
    for r in sel:
        vox_sel[r.get("corpus", "tata")].add(V.voice_key(r))
    L = ["# Base diversa (v4) — relatorio de selecao", "",
         f"Parametros: min_clips={cfg.min_clips}, caps={cfg.caps}, default_cap={cfg.default_cap}, "
         f"holdout_groups(val,test)={cfg.holdout_groups}, heldout_cap={cfg.heldout_cap}, seed={cfg.seed}", "",
         "## Origem -> selecao", "", "| corpus | clipes origem | vozes origem | clipes sel. | vozes sel. |",
         "|---|---|---|---|---|"]
    for c in sorted(by_c_all):
        L.append(f"| {c} | {by_c_all[c]} | {len(vox_all[c])} | {by_c_sel.get(c, 0)} | {len(vox_sel.get(c, ()))} |")
    L += ["", "## Splits (por GRUPO; val/test = vozes nunca vistas no treino)", ""]
    for s in ("train", "val", "test"):
        t = summ[s]["_total"]
        L.append(f"- **{s}**: {t['clips']} clipes, {t['hours']} h, {t['voices']} vozes")
        for c, d in summ[s].items():
            if c != "_total":
                L.append(f"  - {c}: {d['clips']} clipes, {d['hours']} h, {d['voices']} vozes, {d['groups']} grupos")
    L += ["", f"Vazamentos (devem ser vazios): {audit}", "",
          "## Amostragem por massa (default: voz<=1 %, grupo<=2 %, beta=0.5, repeticao<=6x)", "",
          f"- vozes no treino: {info['n_voices']} em {info['n_groups']} grupos",
          f"- **vozes efetivas** (1/soma(m^2)): {info['eff_voices']:.0f}",
          f"- maior massa por voz: {info['max_voice_mass'] * 100:.2f} % | por grupo: "
          f"{info['max_group_mass'] * 100:.2f} % | teto relaxado: {info['relaxed']}",
          f"- maior repeticao esperada de um clipe: {info['max_repeat_obs']:.1f}x",
          "- massa por corpus: " + ", ".join(f"{c}={m * 100:.1f}%" for c, m in sorted(info['corpus_mass'].items())),
          "", "Comparacao: r73_01 tinha 6 locutores CML com ~24 % das amostras e 32 CETUC com ~25 %.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def gender_report(final: list[dict], splits: dict, weights: dict, corpora) -> list[str]:
    """Linhas (markdown) com vozes/clipes/massa de amostragem por genero nos corpora balanceados."""
    by_idx = {r["idx"]: r for r in final}
    L: list[str] = []
    for c in corpora:
        if not any(r.get("corpus") == c for r in final):
            continue
        L += ["", f"## Genero em `{c}` (vozes F = M por construcao)", "",
              "| split | vozes F | vozes M | clipes F | clipes M | massa de amostragem F | massa M |",
              "|---|---|---|---|---|---|---|"]
        for sp in ("train", "val", "test"):
            vf, vm, cf, cm = set(), set(), 0, 0
            mf = mm = 0.0
            for i in splits[sp]:
                r = by_idx[i]
                if r.get("corpus") != c:
                    continue
                g = V.gender_of(r)
                if g == "F":
                    vf.add(r["speaker"]); cf += 1; mf += weights.get(i, 0.0)
                elif g == "M":
                    vm.add(r["speaker"]); cm += 1; mm += weights.get(i, 0.0)
            ms = (f"{mf * 100:.2f} %", f"{mm * 100:.2f} %") if sp == "train" else ("-", "-")
            L.append(f"| {sp} | {len(vf)} | {len(vm)} | {cf} | {cm} | {ms[0]} | {ms[1]} |")
    return L


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=str(paths.TRAINING), help="training/ de origem (tokens, wavs24, meta)")
    ap.add_argument("--out", default=str(paths.ARTIFACTS / "training_v4"))
    ap.add_argument("--min-clips", type=int, default=12)
    ap.add_argument("--cap-tagarela", type=int, default=40)
    ap.add_argument("--cap-podcast", type=int, default=40)
    ap.add_argument("--cap-cetuc", type=int, default=120)
    ap.add_argument("--cap-cml", type=int, default=120)
    ap.add_argument("--cap-tata", type=int, default=120)
    ap.add_argument("--cap-cv", type=int, default=25)
    ap.add_argument("--cap-mls", type=int, default=60)
    ap.add_argument("--no-clip-caps", action="store_true",
                    help="TREINO sem teto de clipes por voz (o equilibrio entre vozes fica so com o amostrador "
                         "por massa); val/test continuam com --heldout-cap. Mais textos/palavras unicos.")
    ap.add_argument("--exclude-speakers", type=Path, default=None,
                    help="arquivo com um locutor por linha a descartar (speaker_purity.py -> exclude_speakers.txt)")
    ap.add_argument("--exclude-clips", type=Path, default=None,
                    help="arquivo com um idx por linha a descartar (speaker_purity.py -> exclude_clips.txt)")
    ap.add_argument("--heldout-cap", type=int, default=24)
    ap.add_argument("--holdout", type=str, default=None,
                    help="JSON {corpus: [n_val_grupos, n_test_grupos]} (default: voices.DEFAULT_HOLDOUT_GROUPS)")
    ap.add_argument("--exclude-corpora", nargs="*", default=[])
    ap.add_argument("--extra-src", nargs="*", default=[],
                    help="pastas de corpora extras (saida de ingest_extra_corpus.py: dataset_meta.jsonl + "
                         "tokens/ + wavs24/); entram na selecao e sao ligadas junto com o resto")
    ap.add_argument("--extra-meta", nargs="*", default=[],
                    help="dataset_meta.jsonl de corpora extras (ex.: Common Voice ja processado em --src)")
    ap.add_argument("--plan-steps", type=int, default=2500,
                    help="passos de otimizador previstos (so p/ o relatorio da amostragem)")
    ap.add_argument("--plan-micro", type=int, default=32, help="amostras por passo (batch*grad_acc)")
    ap.add_argument("--max-repeat", type=float, default=6.0,
                    help="teto de repeticoes por clipe no treino (0 = sem teto)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--link", choices=["hardlink", "copy", "none"], default="none")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    if src.resolve() == out.resolve():
        sys.exit("[erro] --out nao pode ser igual a --src")
    out.mkdir(parents=True, exist_ok=True)

    cfg = V.VoiceCfg(min_clips=args.min_clips, seed=args.seed, heldout_cap=args.heldout_cap,
                     exclude_corpora=tuple(args.exclude_corpora))
    cfg.caps.update({"tagarela": args.cap_tagarela, "podcast": args.cap_podcast, "cetuc": args.cap_cetuc,
                     "cml_pt": args.cap_cml, "tata": args.cap_tata, "cv_pt": args.cap_cv,
                     "mls_pt": args.cap_mls})
    if args.no_clip_caps:
        cfg.caps = {k: 10 ** 9 for k in cfg.caps}
        cfg.default_cap = 10 ** 9
    if args.holdout:
        cfg.holdout_groups.update({k: tuple(v) for k, v in json.loads(args.holdout).items()})

    recs = meta_io.load_meta_records(src / "dataset_meta.jsonl")
    for xs in args.extra_src:
        recs += meta_io.load_meta_records(Path(xs) / "dataset_meta.jsonl")
    for extra in args.extra_meta:
        recs += [json.loads(ln) for ln in Path(extra).read_text(encoding="utf-8").splitlines() if ln.strip()]
    print(f"[meta] {len(recs)} registros de {src}")
    if args.exclude_speakers or args.exclude_clips:
        def _lines(p):
            return {ln.strip() for ln in Path(p).read_text(encoding="utf-8").splitlines() if ln.strip()} if p else set()
        bad_spk, bad_clip = _lines(args.exclude_speakers), _lines(args.exclude_clips)
        n0 = len(recs)
        recs = [r for r in recs if r["idx"] not in bad_clip and r.get("speaker") not in bad_spk]
        print(f"[pureza] excluidos {n0 - len(recs)} clipes ({len(bad_spk)} locutores RUIM, {len(bad_clip)} clipes outliers)")
    sel = V.select_voices(recs, cfg)
    splits = V.make_voice_splits(sel, cfg)
    audit = V.audit(sel, splits)
    if audit["group_leaks"] or audit["speaker_leaks"]:
        sys.exit(f"[erro] vazamento entre splits: {audit}")
    summ = V.split_summary(sel, splits)
    by_idx = {r["idx"]: r for r in sel}
    kept = {i for lst in splits.values() for i in lst}
    final = [by_idx[i] for i in sorted(kept)]
    weights, info = V.speaker_mass_weights(
        [by_idx[i] for i in splits["train"]], total_samples=args.plan_steps * args.plan_micro,
        max_repeat=args.max_repeat)

    with (out / "dataset_meta.jsonl").open("w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(MANIFEST_COLS)
        for r in final:
            w.writerow([r["idx"], r.get("corpus", "tata"), r.get("speaker", ""), r.get("wav_rel", ""),
                        r.get("text", ""), r.get("dur_src_s", ""), r.get("dur_proc_s", ""),
                        r.get("frames", ""), "ref_edit_tata"])
    SPL.write_splits(splits, out)
    (out / "splits_meta.json").write_text(json.dumps(
        {"protocol": "v4: vozes diversas (min_clips, teto de clipes por voz), val/test = grupos inteiros",
         "seed": cfg.seed, "cfg": {"min_clips": cfg.min_clips, "caps": cfg.caps,
                                   "holdout_groups": cfg.holdout_groups, "heldout_cap": cfg.heldout_cap},
         "stats": summ, "audit": audit, "sampler_default": info, "source": str(src)},
        indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    with (out / "speaker_table.csv").open("w", encoding="utf-8", newline="") as f:
        rows = V.voice_table(final, splits)
        w = csv.DictWriter(f, fieldnames=["split", "corpus", "group", "speaker", "n_clips", "hours"])
        w.writeheader()
        w.writerows(rows)
    write_report(out / "selection_report.md", cfg, recs, final, splits, info, summ, audit)
    gl = gender_report(final, splits, weights, cfg.balance_gender)
    if gl:
        with (out / "selection_report.md").open("a", encoding="utf-8") as f:
            f.write("\n".join(gl) + "\n")
        print("\n".join(gl))
    tr = summ["train"]["_total"]
    print(f"[ok] train={tr['clips']} clipes/{tr['hours']} h/{tr['voices']} vozes | "
          f"val={summ['val']['_total']['voices']} vozes | test={summ['test']['_total']['voices']} vozes | "
          f"vozes efetivas (sampler)={info['eff_voices']:.0f}")
    print(f"[ok] relatorio: {out / 'selection_report.md'}")

    if args.link != "none":
        res = link_assets([r["idx"] for r in final], [src, *map(Path, args.extra_src)], out, args.link)
        print(f"[link] {res['stats']}")
        if res["missing"]:
            (out / "missing_assets.txt").write_text("\n".join(res["missing"]), encoding="utf-8")
            sys.exit(f"[erro] {len(res['missing'])} arquivos de origem ausentes (ver missing_assets.txt)")
    else:
        print("[aviso] --link none: tokens/ e wavs24/ NAO foram criados (rode de novo com --link hardlink)")


if __name__ == "__main__":
    main()
