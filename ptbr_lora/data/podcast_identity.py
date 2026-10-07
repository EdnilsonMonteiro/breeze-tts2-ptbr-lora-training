"""podcast_identity.py — reorganiza os locutores do podcast diarizado: 1 rotulo = 1 pessoa, inclusive
entre episodios diferentes. Gera `datasets/podcast/speakers.jsonl` (mesmo padrao dos outros corpora:
idx -> speaker [+ group]) e um relatorio para OUVIR as decisoes.

Etapas (todas registradas por clipe em identity/chunks.csv):
  1. borda   : descarta clipe cuja margem (pad do corte) invade fala exclusiva de OUTRO locutor (RTTM);
  2. troca   : embeddings da 1a e da 2a metade do clipe; se forem pessoas diferentes, descarta;
  3. rotulo  : dentro de cada rotulo do diarizador (episodio:SPEAKER_xx) tira clipes intrusos e separa
               rotulo que contem 2 vozes (vira #a/#b, cada um so se for coeso);
  4. ligacao : rotulos de episodios diferentes que sao a MESMA pessoa viram um id global (podcast:P###),
               por ligacao completa com limiar calibrado em pares sabidamente diferentes (CETUC);
  5. grupos  : episodios que compartilham alguem formam um grupo (unidade do split: a mesma pessoa
               nunca fica em treino e validacao ao mesmo tempo).
Clipes descartados ficam com speaker "podcast:99" (o pipeline de selecao ignora esse sufixo).

Requer o venv do projeto (torch + speechbrain + soundfile + scipy); GPU recomendada (CPU funciona).
Uso:  python ptbr_lora/data/podcast_identity.py [--dry-run] [--device cuda]
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from math import gcd
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "core"), str(_ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import paths  # noqa: E402
import purity as P  # noqa: E402
import speaker_identity as SI  # noqa: E402

SR = 16000
UNASSIGNED = "podcast:99"


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- entrada
def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in Path(p).read_text(encoding="utf-8").splitlines() if ln.strip()]


def read_rttm(d: Path) -> dict[str, list[tuple[float, float, str]]]:
    out: dict[str, list] = defaultdict(list)
    for f in sorted(Path(d).glob("*.rttm")):
        for ln in f.read_text(encoding="utf-8").splitlines():
            p = ln.split()
            if len(p) >= 8 and p[0] == "SPEAKER":
                s = float(p[3])
                out[p[1]].append((s, s + float(p[4]), p[7]))
    return out


# --------------------------------------------------------------------------- audio / embeddings
def load_16k(path: Path, trim_s: float = 0.0) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly

    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != SR:
        g = gcd(int(sr), SR)
        y = resample_poly(y, SR // g, int(sr) // g).astype(np.float32)
    t = int(trim_s * SR)
    if t and len(y) > 2 * t + SR:
        y = y[t:len(y) - t]
    return y


def rms_norm(y: np.ndarray) -> np.ndarray:
    r = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
    return (y * (0.05 / r)).astype(np.float32) if r > 1e-6 else y.astype(np.float32)


class EmbStore:
    """Cache de embeddings em disco (retomavel). chave -> vetor."""

    def __init__(self, path: Path, model: str):
        self.path, self.model, self.d = path, model, {}
        if path.is_file():
            z = np.load(path, allow_pickle=False)
            if str(z["model"]) == model:
                self.d = dict(zip(z["keys"].tolist(), z["emb"]))

    def save(self):
        if self.d:
            tmp = self.path.with_name(self.path.stem + ".tmp.npz")
            ks = list(self.d)
            np.savez(tmp, model=np.array(self.model), keys=np.array(ks), emb=np.stack([self.d[k] for k in ks]))
            os.replace(tmp, self.path)


def embed_jobs(jobs: list[tuple[str, Path, float, str]], store: EmbStore, embedder, batch: int,
               label: str, half_min_s: float, deadline: float = 0.0, save_every: float = 60.0
               ) -> tuple[list[tuple[str, str]], bool]:
    """jobs: (idx, wav, trim_s, modo['full'|'both']). Chaves: idx|full, idx|h1, idx|h2.
    -> (falhas, completo). Com `deadline` (time.time()), para no prazo e salva (rode de novo p/ continuar)."""
    todo = [j for j in jobs if f"{j[0]}|full" not in store.d]
    log(f"[{label}] {len(jobs)} clipes | em cache {len(jobs) - len(todo)} | a calcular {len(todo)}")
    fails: list[tuple[str, str]] = []
    t0 = last = time.time()
    complete = True
    for i in range(0, len(todo), batch):
        if deadline and time.time() > deadline:
            complete = False
            break
        keys, wavs = [], []
        for idx, wav, trim, mode in todo[i:i + batch]:
            try:
                y = load_16k(wav, trim)
            except Exception as exc:  # noqa: BLE001
                fails.append((idx, f"{type(exc).__name__}: {str(exc)[:60]}"))
                continue
            if len(y) < SR:
                fails.append((idx, f"curto ({len(y) / SR:.1f}s)"))
                continue
            keys.append(f"{idx}|full")
            wavs.append(rms_norm(y[:10 * SR]))
            if mode == "both" and len(y) >= 2 * half_min_s * SR:
                h = len(y) // 2
                keys += [f"{idx}|h1", f"{idx}|h2"]
                wavs += [rms_norm(y[:h]), rms_norm(y[h:])]
        if wavs:
            order = np.argsort([len(w) for w in wavs])          # lotes de tamanho parecido
            for k in range(0, len(order), batch):
                sel = order[k:k + batch]
                E = embedder([wavs[j] for j in sel])
                for j, e in zip(sel, E):
                    store.d[keys[j]] = e.astype(np.float32)
        done = min(len(todo), i + batch)
        el = time.time() - t0
        rate = done / max(el, 1e-6)
        sys.stdout.write(f"\r[{label}] {done}/{len(todo)} {done / max(1, len(todo)):5.1%} | {rate:5.1f} clipes/s | "
                         f"resta ~{(len(todo) - done) / max(rate, 1e-6) / 60:4.1f} min   ")
        sys.stdout.flush()
        if time.time() - last > save_every:
            store.save()
            last = time.time()
    if todo:
        sys.stdout.write("\n")
    store.save()
    return fails, complete


# --------------------------------------------------------------------------- relatorio
def _safe(s: str) -> str:
    return re.sub(r"[^\w.-]+", "_", s)[:50]


def write_report(out: Path, summary: dict, speakers: list[dict], examples: list[dict]) -> None:
    e = html.escape
    lst = out / "listen"
    if lst.exists():
        shutil.rmtree(lst)
    lst.mkdir(parents=True)
    cards = []
    for k, ex in enumerate(examples, 1):
        d = lst / f"{k:03d}_{_safe(ex['title'])}"
        d.mkdir()
        items = []
        for j, (lab, src) in enumerate(ex["clips"]):
            dst = d / f"{j}_{_safe(lab)}.wav"
            try:
                shutil.copy2(src, dst)
                items.append(f'<div class="ex"><span>{e(lab)}</span><audio controls preload="none" '
                             f'src="{e(dst.relative_to(out).as_posix())}"></audio></div>')
            except OSError:
                pass
        cards.append(f'<div class="card {ex["kind"]}"><h3>{e(ex["title"])}</h3><p>{e(ex["desc"])}</p>{"".join(items)}</div>')
    rows = "".join(f"<tr><td>{e(s['speaker'])}</td><td>{s['n_clips']}</td><td>{s['hours']:.2f}</td>"
                   f"<td>{s['intra']:.3f}</td><td>{len(s['episodes'])}</td><td>{e(', '.join(s['aliases']))}</td></tr>"
                   for s in speakers)
    st = summary["status"]
    page = f"""<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Locutores do podcast</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{{font-family:system-ui,sans-serif;margin:16px;max-width:1150px;color:#222;background:#fff}}
table{{border-collapse:collapse;font-size:13px}}td,th{{border:1px solid #ccc;padding:3px 7px;text-align:left}}
.card{{border:1px solid #ccc;border-left:6px solid #999;border-radius:6px;padding:8px 12px;margin:10px 0}}
.card.merge{{border-left-color:#2e7d32}}.card.split{{border-left-color:#e6a100}}.card.drop{{border-left-color:#c0392b}}
.ex{{display:inline-flex;flex-direction:column;margin:4px 12px 4px 0;font-size:12px}}audio{{width:230px}}
h3{{margin:4px 0;font-size:15px}}p{{margin:4px 0;font-size:13px}}</style>
<h1>Locutores do podcast</h1>
<p>{summary['n_candidates']} clipes | mantidos {st.get('kept', 0)} ({summary['kept_hours']:.1f} h) |
borda {st.get('drop_bleed', 0)} | troca de voz {st.get('drop_mixed', 0)} | intruso {st.get('drop_outlier', 0)} |
rotulo 2 vozes {st.get('drop_split', 0)} | locutor pequeno {st.get('drop_small', 0)} |
pessoa impura {st.get('drop_impure', 0)}</p>
<p>Rotulos do diarizador: {summary['n_labels']} -> pessoas: <b>{summary['n_global']}</b>
(das quais {summary['n_multi_episode']} aparecem em mais de um episodio) | grupos de split: {summary['n_groups']} |
limiar de ligacao {summary['link_threshold']:.3f}</p>
<h2>Ouvir as decisoes</h2>
<p><b>Verde</b>: a mesma pessoa em episodios diferentes (deve soar igual). <b>Amarelo</b>: rotulo separado em
dois (devem soar diferentes). <b>Vermelho</b>: clipes descartados.</p>
{''.join(cards)}
<h2>Pessoas</h2><table><tr><th>id</th><th>clipes</th><th>horas</th><th>teto</th><th>episodios</th>
<th>rotulos originais</th></tr>{rows}</table></html>"""
    (out / "report.html").write_text(page, encoding="utf-8")


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--index", type=Path, default=paths.SCRAPING_WORK / "chunks_index.jsonl")
    ap.add_argument("--rttm-dir", type=Path, default=paths.SCRAPING_WORK / "rttm")
    ap.add_argument("--meta", type=Path, default=paths.TRAINING / "dataset_meta.jsonl",
                    help="dataset_meta.jsonl com os clipes tokenizados (so esses entram)")
    ap.add_argument("--wavs", type=Path, default=paths.TRAINING / "wavs24")
    ap.add_argument("--dataset-dir", type=Path, default=paths.DATASETS_ROOT / "podcast")
    ap.add_argument("--pad", type=float, default=0.175, help="margem usada no corte (02_slice.py)")
    ap.add_argument("--min-clips", type=int, default=8, help="clipes limpos minimos para virar locutor")
    ap.add_argument("--link-floor", type=float, default=0.75, help="limiar minimo de ligacao entre episodios")
    ap.add_argument("--calib-corpus", default="cetuc")
    ap.add_argument("--calib-speakers", type=int, default=60)
    ap.add_argument("--calib-clips", type=int, default=8)
    ap.add_argument("--half-min-s", type=float, default=1.5, help="metade minima (s) p/ o teste de troca")
    ap.add_argument("--half-margin", type=float, default=0.15,
                    help="troca de voz exige metade x metade abaixo da mediana do rotulo menos isto")
    ap.add_argument("--keep-impure", action="store_true",
                    help="mantem pessoas com pureza RUIM (padrao: descarta; teto < 0,45 ou 2 vozes)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true", help="nao grava speakers.jsonl (so o relatorio)")
    ap.add_argument("--only-tags", default="", help="so estes episodios (virgula) — auditoria rapida")
    ap.add_argument("--out-dir", type=Path, default=None, help="pasta do relatorio (padrao <dataset-dir>/identity)")
    ap.add_argument("--time-budget", type=float, default=0, help="segundos; para, salva o cache e sai com 3")
    ap.add_argument("--save-every", type=float, default=60)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out = args.out_dir or (args.dataset_dir / "identity")
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    # ---------------- candidatos: clipes do indice que existem no meta (tokenizados)
    meta = {r["idx"]: r for r in read_jsonl(args.meta)}
    index = [r for r in read_jsonl(args.index) if r["chunk"].removesuffix(".wav") in meta]
    if args.only_tags:
        keep_tags = {t.strip() for t in args.only_tags.split(",") if t.strip()}
        index = [r for r in index if r["tag"] in keep_tags]
    for r in index:
        r["idx"] = r["chunk"].removesuffix(".wav")
        r["label"] = f"{r['tag']}:{r['speaker']}"
        r["dur"] = float(meta[r["idx"]].get("dur_proc_s", r.get("dur", 0)))
    rttm = read_rttm(args.rttm_dir)
    log(f"[entrada] {len(index)} clipes de podcast tokenizados | {len({r['tag'] for r in index})} episodios | "
        f"{len({r['label'] for r in index})} rotulos do diarizador | rttm de {len(rttm)} episodios")

    # ---------------- embeddings
    import speaker_purity as SPU

    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[modelo] {SPU.MODEL_ID} em {device}")
    embedder = SPU.EcapaEmbedder(device, SPU.default_model_dir(out))
    store = EmbStore(out / "embeddings.npz", SPU.MODEL_ID)
    deadline = time.time() + args.time_budget if args.time_budget else 0.0
    fails, ok1 = embed_jobs([(r["idx"], args.wavs / f"{r['idx']}.wav", args.pad, "both") for r in index],
                            store, embedder, args.batch, "podcast", args.half_min_s, deadline, args.save_every)

    # calibracao: pessoas sabidamente diferentes (e iguais) de outro corpus
    by_spk = defaultdict(list)
    for r in meta.values():
        if r.get("corpus") == args.calib_corpus and r.get("speaker") and 3.0 <= float(r.get("dur_proc_s", 0)) <= 10:
            by_spk[r["speaker"]].append(r["idx"])
    cal_spk = sorted(by_spk, key=lambda s: SI_hash(s))[:args.calib_speakers]
    cal = [(i, args.wavs / f"{i}.wav", 0.0, "both") for s in cal_spk for i in sorted(by_spk[s], key=SI_hash)[:args.calib_clips]]
    f2, ok2 = embed_jobs(cal, store, embedder, args.batch, "calibracao", args.half_min_s, deadline, args.save_every)
    fails += f2
    if not (ok1 and ok2):
        log("[parcial] prazo atingido; cache salvo — rode de novo para continuar")
        return 3
    cal_groups = [np.stack([store.d[f"{i}|full"] for i in sorted(by_spk[s], key=SI_hash)[:args.calib_clips]
                            if f"{i}|full" in store.d]) for s in cal_spk]
    cal_groups = [g for g in cal_groups if len(g) >= 4]
    diff = SI.centroid_pair_sims(cal_groups) if len(cal_groups) >= 2 else np.array([])
    same = SI.split_half_sims(cal_groups)
    thr = SI.calibrate_link_threshold(diff, floor=args.link_floor)
    cal_half = np.array([float(store.d[f"{i}|h1"] @ store.d[f"{i}|h2"]) for i, *_ in cal
                         if f"{i}|h1" in store.d])
    half_thr = float(np.clip(np.percentile(cal_half, 1) if cal_half.size else 0.3, 0.15, 0.45))
    log(f"[calibracao] {len(cal_groups)} pessoas de {args.calib_corpus}: centroides de pessoas DIFERENTES "
        f"p99.9={np.percentile(diff, 99.9) if diff.size else float('nan'):.3f} max={diff.max() if diff.size else float('nan'):.3f} | "
        f"MESMA pessoa (metades) p1={np.percentile(same, 1) if same.size else float('nan'):.3f} -> limiar de ligacao {thr:.3f}")
    log(f"[calibracao] metades do MESMO clipe (mesma pessoa) p1={half_thr:.3f} -> abaixo disso = troca de voz")

    # ---------------- 1+2: borda e troca de voz
    status: dict[str, str] = {}
    half_sim: dict[str, float] = {}
    for r in index:
        i = r["idx"]
        if f"{i}|full" not in store.d:
            status[i] = "drop_fail"
            continue
        if SI.edge_bleed(r["spans"], args.pad, r["speaker"], rttm.get(r["tag"], [])):
            status[i] = "drop_bleed"
            continue
        if f"{i}|h1" in store.d:
            half_sim[i] = float(store.d[f"{i}|h1"] @ store.d[f"{i}|h2"])
    # troca de voz: metade x metade abaixo do limiar calibrado E bem abaixo do tipico do proprio rotulo
    # (canal ruim -- convidado por telefone -- baixa a similaridade de TODAS as metades; isso nao e troca)
    med_half = {}
    for r in index:
        if r["idx"] in half_sim and r["idx"] not in status:
            med_half.setdefault(r["label"], []).append(half_sim[r["idx"]])
    med_half = {k: float(np.median(v)) for k, v in med_half.items()}
    for r in index:
        i = r["idx"]
        if i in status or i not in half_sim:
            continue
        m = med_half.get(r["label"], 1.0)
        if half_sim[i] < half_thr and (half_sim[i] < m - args.half_margin or m < half_thr):
            status[i] = "drop_mixed"

    # ---------------- 3: limpeza por rotulo
    cfg = P.PurityConfig()
    by_label = defaultdict(list)
    for r in index:
        if r["idx"] not in status:
            by_label[r["label"]].append(r["idx"])
    sub_of: dict[str, str] = {}
    label_notes = {}
    subs: dict[str, list[str]] = {}
    for lab in sorted(by_label):
        ids = by_label[lab]
        E = np.stack([store.d[f"{i}|full"] for i in ids])
        rf = SI.refine_label(lab, ids, E, cfg, min_sub_clips=args.min_clips)
        for cid, why in rf.dropped.items():
            status[cid] = "drop_split" if "2 vozes" in why else "drop_outlier"
        for sub, cids in rf.kept.items():
            if len(cids) < args.min_clips:
                for c in cids:
                    status[c] = "drop_small"
                continue
            subs[sub] = cids
            for c in cids:
                sub_of[c] = sub
        if rf.note:
            label_notes[lab] = rf.note

    # ---------------- 4: ligacao entre episodios
    names = sorted(subs)
    C = np.stack([P.l2n(np.stack([store.d[f"{c}|full"] for c in subs[n]]).mean(axis=0)) for n in names])
    lab = SI.complete_linkage(C, thr) if len(names) > 1 else np.zeros(len(names), dtype=int)
    hours = {n: sum(meta[c]["dur_proc_s"] for c in subs[n]) / 3600 for n in names}
    clusters = defaultdict(list)
    for n, k in zip(names, lab):
        clusters[int(k)].append(n)
    order = sorted(clusters, key=lambda k: -sum(hours[n] for n in clusters[k]))
    gid_of_sub = {}
    for rank, k in enumerate(order, 1):
        for n in clusters[k]:
            gid_of_sub[n] = f"podcast:P{rank:03d}"
    members = defaultdict(list)
    for n, g in gid_of_sub.items():
        members[g].append(n.split(":")[0])
    group_of_ep = SI.episode_groups(members)
    for r in index:                                    # episodios sem ninguem mantido -> grupo proprio
        group_of_ep.setdefault(r["tag"], f"E_{r['tag']}")

    # ---------------- resultados
    rows, spk_rows = [], []
    for r in index:
        i = r["idx"]
        st = status.get(i, "kept")
        g = gid_of_sub.get(sub_of.get(i, ""), "")
        if st == "kept" and not g:
            st = "drop_small"
        rows.append({"idx": i, "tag": r["tag"], "label": r["label"], "sub": sub_of.get(i, ""),
                     "speaker": g if st == "kept" else UNASSIGNED, "group": group_of_ep[r["tag"]],
                     "status": st, "half_sim": round(half_sim.get(i, float("nan")), 4), "dur": r["dur"]})
    kept = [x for x in rows if x["status"] == "kept"]
    by_g = defaultdict(list)
    for x in kept:
        by_g[x["speaker"]].append(x)
    for g in sorted(by_g):
        xs = by_g[g]
        E = np.stack([store.d[f"{x['idx']}|full"] for x in xs])
        rr = P.analyze_speaker(g, [x["idx"] for x in xs], E, cfg)
        spk_rows.append({"speaker": g, "n_clips": len(xs), "hours": sum(x["dur"] for x in xs) / 3600,
                         "intra": rr.intra, "episodes": sorted({x["tag"] for x in xs}),
                         "aliases": sorted({x["sub"] for x in xs}), "group": xs[0]["group"],
                         "status_purity": rr.status})
    if not args.keep_impure:
        bad = {s["speaker"] for s in spk_rows if s["status_purity"] == "RUIM"}
        for x in rows:
            if x["speaker"] in bad:
                x["status"], x["speaker"] = "drop_impure", UNASSIGNED
        spk_rows = [s for s in spk_rows if s["speaker"] not in bad]
        kept = [x for x in rows if x["status"] == "kept"]
    st_count = Counter(x["status"] for x in rows)
    summary = {
        "n_candidates": len(rows), "status": dict(st_count),
        "kept_hours": sum(x["dur"] for x in kept) / 3600,
        "n_labels": len({r["label"] for r in index}), "n_global": len(spk_rows),
        "n_multi_episode": sum(len(s["episodes"]) > 1 for s in spk_rows),
        "n_groups": len({s["group"] for s in spk_rows}),
        "link_threshold": thr, "half_threshold": half_thr,
        "calib": {"corpus": args.calib_corpus, "n_people": len(cal_groups),
                  "diff_p99_9": float(np.percentile(diff, 99.9)) if diff.size else None,
                  "diff_max": float(diff.max()) if diff.size else None,
                  "same_p1": float(np.percentile(same, 1)) if same.size else None},
        "label_notes": label_notes, "fails": fails[:50], "n_fails": len(fails),
        "minutes": round((time.time() - t_all) / 60, 1)}

    with open(out / "chunks.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out / "speakers.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["speaker", "group", "n_clips", "hours", "intra", "purity", "n_episodes", "episodes", "aliases"])
        for s in spk_rows:
            w.writerow([s["speaker"], s["group"], s["n_clips"], f"{s['hours']:.3f}", f"{s['intra']:.3f}",
                        s["status_purity"], len(s["episodes"]), " ".join(s["episodes"]), " ".join(s["aliases"])])
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    # exemplos para ouvir: ligacoes entre episodios, rotulos separados, descartes
    wav = lambda i: args.wavs / f"{i}.wav"  # noqa: E731
    ex = []
    for s in sorted(spk_rows, key=lambda s: -len(s["episodes"])):
        if len(s["episodes"]) < 2:
            continue
        clips = []
        for a in s["aliases"][:4]:
            c = next(x["idx"] for x in by_g[s["speaker"]] if x["sub"] == a)
            clips.append((a, wav(c)))
        ex.append({"kind": "merge", "title": f"{s['speaker']} = {len(s['aliases'])} rotulos",
                   "desc": f"{len(s['episodes'])} episodios, teto {s['intra']:.2f}: devem soar como a MESMA pessoa",
                   "clips": clips})
    for labl, note in sorted(label_notes.items())[:12]:
        a = [x for x in rows if x["sub"] == labl + "#a"][:2]
        b = [x for x in rows if x["sub"] == labl + "#b"][:2]
        if a and b:
            ex.append({"kind": "split", "title": f"{labl} separado", "desc": note + ": #a e #b devem soar DIFERENTES",
                       "clips": [("#a", wav(x["idx"])) for x in a] + [("#b", wav(x["idx"])) for x in b]})
    for stt, desc in (("drop_mixed", "troca de voz dentro do clipe"), ("drop_bleed", "margem invade outro locutor"),
                      ("drop_outlier", "destoa do proprio locutor")):
        xs = [x for x in rows if x["status"] == stt][:4]
        if xs:
            ex.append({"kind": "drop", "title": f"descartados: {desc}", "desc": f"{st_count[stt]} clipes",
                       "clips": [(x["label"], wav(x["idx"])) for x in xs]})
    write_report(out, summary, spk_rows, ex[:40])

    if not args.dry_run:
        dst = args.dataset_dir / "speakers.jsonl"
        if dst.exists():
            shutil.copy2(dst, dst.with_suffix(f".jsonl.bak{int(time.time())}"))
        tmp = dst.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for x in rows:
                f.write(json.dumps({"idx": x["idx"], "speaker": x["speaker"], "group": x["group"],
                                    "status": x["status"]}, ensure_ascii=False) + "\n")
        os.replace(tmp, dst)
        log(f"[ok] {dst} ({len(rows)} clipes)")
    log(f"[resumo] mantidos {st_count['kept']} clipes ({summary['kept_hours']:.1f} h) de {len(rows)} | "
        + " | ".join(f"{k}={v}" for k, v in sorted(st_count.items()) if k != "kept"))
    log(f"[resumo] {summary['n_labels']} rotulos -> {summary['n_global']} pessoas "
        f"({summary['n_multi_episode']} em >1 episodio) | {summary['n_groups']} grupos | {summary['minutes']} min")
    log(f"[ok] relatorio: {out / 'report.html'}")
    return 0


def SI_hash(s: str) -> str:
    import hashlib

    return hashlib.sha1(f"42|{s}".encode()).hexdigest()


if __name__ == "__main__":
    sys.exit(main())
