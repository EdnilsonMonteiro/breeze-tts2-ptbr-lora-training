"""speaker_purity.py — auditoria de PUREZA DE LOCUTOR de um dataset de voz.

Verifica, com embeddings de locutor (ECAPA), se os clipes de cada rotulo de locutor sao mesmo
a mesma pessoa. Serve para qualquer dataset (corpus publico, podcast diarizado, audio proprio)
e roda ANTES de treinar: rotulo errado ensina o modelo de clonagem a ignorar a referencia.

Gera em --out:
  report.html          resumo + exemplos para OUVIR (clipe tipico x clipe que destoa)
  report.md            o mesmo resumo em texto
  speakers.csv         uma linha por locutor: teto, outliers, teste de 2 vozes, mais parecido, status
  clips.csv            uma linha por clipe: similaridade com o proprio locutor, outlier?
  duplicates.csv       pares de rotulos que parecem a MESMA pessoa (cross_split=1 -> vazamento)
  exclude_speakers.txt locutores RUIM (um por linha)
  exclude_clips.txt    clipes outliers de locutores nao-RUIM (um por linha)
  summary.json         numeros por corpus/grupo + configuracao usada
  embeddings.npz       cache (rodar de novo so calcula o que falta)

Entradas (escolha uma):
  --training-dir DIR   preset do projeto: DIR/manifest.csv + speaker_table.csv + wavs24/<idx>.wav
  --manifest ARQ       CSV/TSV/JSONL qualquer + modelos de campo:
                       --audio "{wav_rel}" --speaker "{speaker}" [--group "{corpus}"] [--split "{split}"]
                       [--id "{idx}"] [--root DIR_BASE_DOS_AUDIOS]
  --folders DIR        DIR/<locutor>/**/*.wav  (o jeito mais simples para audio proprio)

Exemplos:
  python ptbr_lora/tools/speaker_purity.py --training-dir <ARTIFACTS>/training_v4 --out purity_v4
  python ptbr_lora/tools/speaker_purity.py --manifest work/chunks_index.jsonl --root datasets/podcast/wavs \
         --audio "{chunk}" --speaker "{tag}:{speaker}" --group "{tag}" --out purity_podcast
  python ptbr_lora/tools/speaker_purity.py --folders meus_audios --out purity_meus
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
from math import gcd
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CORE = _HERE.parents[0] / "core"
for _p in (str(_CORE), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import purity as P  # noqa: E402

SR = 16000
MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"


# --------------------------------------------------------------------------- audio
def load_audio_16k(path: Path, max_s: float) -> np.ndarray:
    import soundfile as sf

    try:
        y, sr = sf.read(str(path), dtype="float32", always_2d=True)
        y = y.mean(axis=1)
    except Exception:  # noqa: BLE001  (mp3/m4a sem suporte no libsndfile)
        import librosa

        y, sr = librosa.load(str(path), sr=None, mono=True)
    if sr != SR:
        try:
            from scipy.signal import resample_poly

            g = gcd(int(sr), SR)
            y = resample_poly(y, SR // g, int(sr) // g).astype(np.float32)
        except ImportError:
            import librosa

            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    if max_s > 0 and len(y) > max_s * SR:                     # trecho central (mais estavel)
        n = int(max_s * SR)
        a = (len(y) - n) // 2
        y = y[a:a + n]
    r = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
    if r > 1e-6:
        y = y * (0.05 / r)                                      # ECAPA e sensivel ao nivel
    return np.asarray(y, dtype=np.float32)


class EcapaEmbedder:
    name = MODEL_ID

    def __init__(self, device: str, savedir: Path):
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier
        self.device = device
        # modelo ja baixado (pasta com hyperparams.yaml) carrega OFFLINE; senao baixa do Hugging Face
        local = (Path(savedir) / "hyperparams.yaml").is_file()
        self.clf = EncoderClassifier.from_hparams(source=str(savedir) if local else MODEL_ID, savedir=str(savedir),
                                                  run_opts={"device": device})

    def __call__(self, wavs: list[np.ndarray]) -> np.ndarray:
        import torch

        n = max(len(w) for w in wavs)
        x = np.zeros((len(wavs), n), dtype=np.float32)
        for i, w in enumerate(wavs):
            x[i, :len(w)] = w
        lens = torch.tensor([len(w) / n for w in wavs], dtype=torch.float32)
        with torch.no_grad():
            e = self.clf.encode_batch(torch.from_numpy(x).to(self.device), lens.to(self.device))
        return P.l2n(e.squeeze(1).detach().cpu().numpy())


def default_model_dir(out: Path) -> Path:
    """Reaproveita o ECAPA ja baixado pelo projeto (dataScrapping/work/spkrec-ecapa), se houver."""
    try:
        import paths

        d = paths.SCRAPING_WORK / "spkrec-ecapa"
        if (d / "hyperparams.yaml").is_file():
            return d
    except Exception:  # noqa: BLE001
        pass
    return out / "_model_ecapa"


# --------------------------------------------------------------------------- cache
def cache_key(c: P.Clip) -> str:
    return f"{c.id}|{c.audio}"


def load_cache(path: Path, model: str) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    z = np.load(path, allow_pickle=False)
    if str(z["model"]) != model:
        print(f"[cache] {path.name} e de outro modelo ({z['model']}); ignorando")
        return {}
    return {k: v for k, v in zip(z["keys"].tolist(), z["emb"])}


def save_cache(path: Path, model: str, cache: dict[str, np.ndarray]) -> None:
    if not cache:
        return
    keys = list(cache)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, model=np.array(model), keys=np.array(keys), emb=np.stack([cache[k] for k in keys]))
    os.replace(tmp, path)


def embed_all(clips: list[P.Clip], embedder, cache_path: Path, batch: int, max_s: float,
              min_s: float) -> tuple[dict[str, np.ndarray], list[tuple[str, str]]]:
    cache = load_cache(cache_path, embedder.name)
    todo = [c for c in clips if cache_key(c) not in cache]
    print(f"[emb] {len(clips)} clipes | ja em cache: {len(clips) - len(todo)} | a calcular: {len(todo)}",
          flush=True)
    failed: list[tuple[str, str]] = []
    t0 = time.time()
    last_save = time.time()
    done = 0
    for i in range(0, len(todo), batch):
        part, wavs = [], []
        for c in todo[i:i + batch]:
            try:
                y = load_audio_16k(c.audio, max_s)
            except Exception as exc:  # noqa: BLE001
                failed.append((c.id, f"{type(exc).__name__}: {str(exc)[:80]}"))
                continue
            if len(y) < min_s * SR:
                failed.append((c.id, f"curto ({len(y) / SR:.2f}s)"))
                continue
            part.append(c)
            wavs.append(y)
        if wavs:
            E = embedder(wavs)
            for c, e in zip(part, E):
                cache[cache_key(c)] = e.astype(np.float32)
        done = min(len(todo), i + batch)
        el = time.time() - t0
        rate = done / max(el, 1e-6)
        eta = (len(todo) - done) / max(rate, 1e-6)
        pct = done / max(1, len(todo))
        bar = "#" * int(pct * 30)
        sys.stdout.write(f"\r[emb] [{bar:<30}] {done}/{len(todo)} {pct:5.1%} | {rate:5.1f} clipes/s | "
                         f"resta ~{eta / 60:4.1f} min | falhas {len(failed)}   ")
        sys.stdout.flush()
        if time.time() - last_save > 60:
            save_cache(cache_path, embedder.name, cache)
            last_save = time.time()
    if todo:
        sys.stdout.write("\n")
    save_cache(cache_path, embedder.name, cache)
    return {c.id: cache[cache_key(c)] for c in clips if cache_key(c) in cache}, failed


# --------------------------------------------------------------------------- relatorios
def _f(x, nd=3):
    return "" if x != x else f"{x:.{nd}f}"


def write_tables(out: Path, results, dups, clips_by_id: dict[str, P.Clip]) -> None:
    with open(out / "speakers.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["speaker", "group", "split", "status", "n_clips", "intra", "loo_median", "loo_p10",
                    "n_outliers", "outlier_frac", "two_voices", "split_sizes", "split_inter",
                    "nearest", "nearest_sim", "reasons"])
        for r in sorted(results, key=lambda r: (r.group, r.speaker)):
            w.writerow([r.speaker, r.group, r.split, r.status, r.n, _f(r.intra), _f(r.loo_median),
                        _f(r.loo_p10), r.n_outliers, _f(r.outlier_frac), int(r.split_suspect),
                        f"{r.split_sizes[0]}+{r.split_sizes[1]}", _f(r.split_inter), r.nearest,
                        _f(r.nearest_sim), "; ".join(r.reasons)])
    with open(out / "clips.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "speaker", "group", "audio", "loo_sim", "outlier", "cluster"])
        for r in results:
            for cid, s, o, k in r.clip_loo:
                w.writerow([cid, r.speaker, r.group, str(clips_by_id[cid].audio), _f(s), int(o), k])
    with open(out / "duplicates.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        cols = ["speaker_a", "speaker_b", "sim", "group_a", "group_b", "split_a", "split_b", "cross_split"]
        w.writerow(cols)
        for d in dups:
            w.writerow([int(d[c]) if isinstance(d[c], bool) else d[c] for c in cols])
    bad = sorted(r.speaker for r in results if r.status == "RUIM")
    (out / "exclude_speakers.txt").write_text("\n".join(bad) + ("\n" if bad else ""), encoding="utf-8")
    outl = sorted(cid for r in results if r.status != "RUIM" for cid, _, o, _ in r.clip_loo if o)
    (out / "exclude_clips.txt").write_text("\n".join(outl) + ("\n" if outl else ""), encoding="utf-8")


def _safe(s: str) -> str:
    return re.sub(r"[^\w.-]+", "_", s)[:60]


def pick_examples(r, clips_by_id) -> list[tuple[str, str]]:
    """[(rotulo, clip_id)] para ouvir: 2 tipicos + ate 2 que destoam (ou 2 de cada grupo se '2 vozes')."""
    srt = sorted(r.clip_loo, key=lambda t: -t[1])
    if r.split_suspect:                                   # 2 de cada grupo: compare A com B
        ex = [(f"grupo {'AB'[k]}", t[0]) for k in (0, 1) for t in [t for t in srt if t[3] == k][:2]]
    else:
        ex = [("tipico", t[0]) for t in srt[:2]]
        ex += [("destoa", t[0]) for t in sorted(r.clip_loo, key=lambda t: t[1])[:2] if t[2] or t[1] < r.loo_p10]
    seen, out = set(), []
    for lab, cid in ex:
        if cid not in seen and cid in clips_by_id:
            seen.add(cid)
            out.append((lab, cid))
    return out


def copy_examples(out: Path, worst, clips_by_id) -> dict[str, list[tuple[str, str, str]]]:
    ldir = out / "listen"
    if ldir.exists():
        shutil.rmtree(ldir)
    res = {}
    for k, r in enumerate(worst, 1):
        d = ldir / f"{k:03d}_{_safe(r.speaker)}"
        d.mkdir(parents=True, exist_ok=True)
        items = []
        for j, (lab, cid) in enumerate(pick_examples(r, clips_by_id)):
            src = clips_by_id[cid].audio
            dst = d / f"{j}_{_safe(lab)}{src.suffix.lower()}"
            try:
                shutil.copy2(src, dst)
                sim = next(s for c, s, _, _ in r.clip_loo if c == cid)
                items.append((lab, dst.relative_to(out).as_posix(), f"{sim:.2f}"))
            except OSError:
                pass
        res[r.speaker] = items
    return res


STATUS_HELP = [
    ("Teto (intra)", "media da similaridade entre pares de clipes do mesmo rotulo. E o quanto a 'mesma voz' se parece "
                     "consigo mesma. Estudio limpo: ~0,65-0,80. Abaixo de ~0,45 raramente e uma pessoa so."),
    ("Destoa (outlier)", "clipe muito abaixo da mediana do proprio locutor: provavel outra pessoa, vinheta ou "
                         "fala sobreposta."),
    ("Parece 2 vozes", "o rotulo se divide em dois grupos coesos e diferentes entre si: tipico de diarizacao que "
                       "juntou duas pessoas."),
    ("Igual a X", "dois rotulos com a mesma voz (ex.: o mesmo apresentador em episodios diferentes). Se estiverem "
                  "em splits diferentes, e VAZAMENTO entre treino e validacao."),
]


def write_reports(out: Path, args, cfg, results, dups, summ, failed, listen, n_clips) -> None:
    worst = [r for r in results if r.status in ("RUIM", "SUSPEITO")]
    worst.sort(key=lambda r: (r.status != "RUIM", r.intra if r.intra == r.intra else 9))
    # ---------------- markdown
    L = ["# Pureza de locutor", "",
         f"Entrada: `{args.source}` | modelo: `{MODEL_ID}` | clipes analisados: {n_clips} "
         f"(ate {args.max_per_speaker} por locutor) | falhas de leitura: {len(failed)}", "",
         "## Por corpus / grupo", "",
         "| grupo | locutores | teto mediano | teto p10 | OK | SUSPEITO | RUIM | clipes que destoam |",
         "|---|---|---|---|---|---|---|---|"]
    for g, s in summ.items():
        L.append(f"| {g or '-'} | {s['speakers']} | {s['intra_median']:.3f} | {s['intra_p10']:.3f} | {s['ok']} | "
                 f"{s['suspeito']} | {s['ruim']} | {s['outlier_clips_frac']:.1%} |")
    L += ["", "Compare o **teto mediano** entre grupos: um corpus bem abaixo dos outros tem rotulos menos confiaveis.", "",
          f"## Piores locutores ({len(worst)} RUIM/SUSPEITO)", "",
          "| status | locutor | grupo | split | clipes | teto | destoam | motivos |", "|---|---|---|---|---|---|---|---|"]
    for r in worst[:60]:
        L.append(f"| {r.status} | {r.speaker} | {r.group} | {r.split} | {r.n} | {_f(r.intra, 2)} | "
                 f"{r.n_outliers} | {'; '.join(r.reasons)} |")
    if dups:
        L += ["", f"## Possiveis duplicatas ({len(dups)}; {sum(d['cross_split'] for d in dups)} entre splits)", "",
              "| a | b | sim | splits |", "|---|---|---|---|"]
        for d in dups[:40]:
            L.append(f"| {d['speaker_a']} | {d['speaker_b']} | {d['sim']:.2f} | {d['split_a']}/{d['split_b']}"
                     f"{' **VAZAMENTO**' if d['cross_split'] else ''} |")
    L += ["", "## Como ler", ""] + [f"- **{a}**: {b}" for a, b in STATUS_HELP]
    L += ["", "Limiares (dependem do modelo de embedding; ajuste por CLI):", "",
          "```json", json.dumps(cfg.to_dict(), indent=1), "```"]
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---------------- html
    e = html.escape
    rows = "".join(
        f"<tr><td>{e(g or '-')}</td><td>{s['speakers']}</td><td><b>{s['intra_median']:.3f}</b></td>"
        f"<td>{s['intra_p10']:.3f}</td><td>{s['ok']}</td><td>{s['suspeito']}</td><td>{s['ruim']}</td>"
        f"<td>{s['outlier_clips_frac']:.1%}</td></tr>" for g, s in summ.items())
    cards = []
    for r in worst:
        if r.speaker not in listen:
            continue
        aud = "".join(f'<div class="ex"><span>{e(lab)} <small>(sim {sim})</small></span>'
                      f'<audio controls preload="none" src="{e(rel)}"></audio></div>'
                      for lab, rel, sim in listen[r.speaker])
        cards.append(f'<div class="card {r.status.lower()}"><h3>{e(r.status)} — {e(r.speaker)}</h3>'
                     f'<p>{e(r.group)} {e(r.split)} | {r.n} clipes | teto {_f(r.intra, 2)} | '
                     f'{e("; ".join(r.reasons))}</p>{aud}</div>')
    help_html = "".join(f"<li><b>{e(a)}</b>: {e(b)}</li>" for a, b in STATUS_HELP)
    page = f"""<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Pureza de locutor</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{{font-family:system-ui,sans-serif;margin:16px;max-width:1100px;color:#222;background:#fff}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:4px 8px;text-align:right}}
td:first-child,th:first-child{{text-align:left}}
.card{{border:1px solid #ccc;border-left:6px solid #999;border-radius:6px;padding:8px 12px;margin:10px 0}}
.card.ruim{{border-left-color:#c0392b}}.card.suspeito{{border-left-color:#e6a100}}
.ex{{display:inline-flex;flex-direction:column;margin:4px 12px 4px 0;font-size:13px}}
audio{{width:240px}}h3{{margin:4px 0;font-size:16px}}p{{margin:4px 0;font-size:13px}}
</style>
<h1>Pureza de locutor</h1>
<p>Entrada: <code>{e(str(args.source))}</code> | modelo <code>{MODEL_ID}</code> | {n_clips} clipes |
{len(results)} locutores | falhas {len(failed)} | duplicatas {len(dups)}
({sum(d['cross_split'] for d in dups)} entre splits)</p>
<h2>Por corpus / grupo</h2>
<table><tr><th>grupo</th><th>locutores</th><th>teto mediano</th><th>teto p10</th><th>OK</th><th>SUSPEITO</th>
<th>RUIM</th><th>clipes que destoam</th></tr>{rows}</table>
<h2>Ouvir os piores ({len(cards)})</h2>
<p>Ouca o <i>tipico</i> e depois o que <i>destoa</i> (ou os grupos A e B): se forem pessoas diferentes, o rotulo esta errado.</p>
{''.join(cards) or '<p>Nenhum locutor RUIM/SUSPEITO.</p>'}
<h2>Como ler</h2><ul>{help_html}</ul>
<p>Tabelas completas: speakers.csv, clips.csv, duplicates.csv. Listas prontas para filtrar: exclude_speakers.txt,
exclude_clips.txt.</p></html>"""
    (out / "report.html").write_text(page, encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(
        {"source": str(args.source), "model": MODEL_ID, "n_clips": n_clips, "n_speakers": len(results),
         "failed": failed[:200], "n_failed": len(failed), "groups": summ, "config": cfg.to_dict(),
         "status": {s: sum(r.status == s for r in results) for s in ("OK", "SUSPEITO", "RUIM", "POUCOS")},
         "duplicates": len(dups), "cross_split_duplicates": sum(d["cross_split"] for d in dups)},
        ensure_ascii=False, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Auditoria de pureza de locutor (os clipes de cada rotulo sao a mesma pessoa?)",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("Exemplos:")[1])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--training-dir", type=Path, help="pasta de treino do projeto (manifest.csv + wavs24/)")
    src.add_argument("--manifest", type=Path, help="CSV/TSV/JSONL com uma linha por clipe")
    src.add_argument("--folders", type=Path, help="pasta com uma subpasta por locutor")
    ap.add_argument("--audio", default="{audio}", help="modelo do caminho do audio (campos da linha entre {})")
    ap.add_argument("--speaker", default="{speaker}", help="modelo do rotulo de locutor, ex. '{tag}:{speaker}'")
    ap.add_argument("--group", default="", help="modelo do grupo/corpus para o resumo, ex. '{corpus}'")
    ap.add_argument("--split", default="", help="modelo do split (train/val/test) p/ detectar vazamento")
    ap.add_argument("--id", default="", help="modelo do id do clipe (padrao: nome do arquivo)")
    ap.add_argument("--root", type=Path, default=None, help="base dos caminhos relativos (padrao: pasta do manifest)")
    ap.add_argument("--out", type=Path, required=True, help="pasta de saida (relatorios + cache)")
    ap.add_argument("--only-group", default="", help="analisa so estes grupos (separados por virgula)")
    ap.add_argument("--max-per-speaker", type=int, default=30, help="clipes sorteados por locutor (0 = todos)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=8.0, help="usa no maximo N s do centro de cada clipe")
    ap.add_argument("--min-seconds", type=float, default=1.0, help="ignora clipes mais curtos que isto")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--listen", type=int, default=25, help="quantos piores locutores com exemplos para ouvir")
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="pasta local do ECAPA (padrao: a do projeto se existir, senao <out>/_model_ecapa)")
    th = ap.add_argument_group("limiares (calibrados p/ ECAPA; mude so se trocar o modelo)")
    for k, v in P.PurityConfig().to_dict().items():
        th.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v, metavar=type(v).__name__.upper(),
                        help=f"padrao {v}")
    return ap


def collect(args) -> list[P.Clip]:
    if args.training_dir:
        args.source = args.training_dir
        return P.clips_from_training_dir(args.training_dir)
    if args.folders:
        args.source = args.folders
        return P.clips_from_folders(args.folders)
    args.source = args.manifest
    return P.clips_from_manifest(args.manifest, audio=args.audio, speaker=args.speaker, group=args.group,
                                 split=args.split, clip_id=args.id, root=args.root)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = P.PurityConfig(**{k: getattr(args, k) for k in P.PurityConfig().to_dict()})
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    clips = collect(args)
    if args.only_group:
        keep = {g.strip() for g in args.only_group.split(",") if g.strip()}
        clips = [c for c in clips if c.group in keep]
    if not clips:
        print("[erro] nenhum clipe encontrado — confira a entrada e os modelos de campo (--audio/--speaker)")
        return 2
    missing = [c for c in clips[:200] if not c.audio.is_file()]
    if len(missing) == min(200, len(clips)):
        print(f"[erro] nenhum audio existe; exemplo de caminho montado: {missing[0].audio}")
        return 2
    n_spk = len({c.speaker for c in clips})
    sel = P.sample_per_speaker(clips, args.max_per_speaker, args.seed)
    print(f"[entrada] {len(clips)} clipes, {n_spk} locutores -> analisando {len(sel)} "
          f"(ate {args.max_per_speaker}/locutor)", flush=True)

    device = args.device
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    print(f"[modelo] {MODEL_ID} em {device}", flush=True)
    embedder = EcapaEmbedder(device, args.model_dir or default_model_dir(out))
    t0 = time.time()
    emb, failed = embed_all(sel, embedder, out / "embeddings.npz", args.batch, args.max_seconds, args.min_seconds)
    print(f"[emb] pronto em {(time.time() - t0) / 60:.1f} min ({len(emb)} ok, {len(failed)} falhas)", flush=True)

    results, dups, summ = P.analyze(sel, emb, cfg)
    by_id = {c.id: c for c in sel}
    write_tables(out, results, dups, by_id)
    worst = sorted([r for r in results if r.status in ("RUIM", "SUSPEITO")],
                   key=lambda r: (r.status != "RUIM", r.intra if r.intra == r.intra else 9))[:args.listen]
    listen = copy_examples(out, worst, by_id) if args.listen > 0 else {}
    write_reports(out, args, cfg, results, dups, summ, failed, listen, len(emb))

    print("\n[resumo por grupo] teto mediano | OK / SUSPEITO / RUIM | clipes que destoam")
    for g, s in summ.items():
        print(f"  {g or '-':<14} {s['intra_median']:.3f} | {s['ok']:>4} / {s['suspeito']:>4} / {s['ruim']:>4} | "
              f"{s['outlier_clips_frac']:.1%}")
    print(f"[duplicatas] {len(dups)} pares ({sum(d['cross_split'] for d in dups)} entre splits)")
    print(f"[ok] abra {out / 'report.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
