"""ingest_extra_corpus.py — traz MAIS VOZES (Common Voice pt, MLS pt, ou qualquer lista audio/texto/locutor)
para o pipeline, no formato do `prepare_dataset` (wavs24/ + tokens/ + dataset_meta.jsonl), numa pasta
PROPRIA (nao toca no `training/`). Depois:  build_diverse_set.py --extra-src <pasta> --link hardlink

Formatos (--format):
  cvhf Common Voice baixado DIRETO do repo do Hugging Face (fsicoli/common_voice_19_0; sem `datasets`, que
       a partir da v4 nao executa scripts de carga): baixa so os TSVs, escolhe vozes (>= min clipes) com
       genero F/M, baixa os tars de audio (train/dev/test), extrai SO os clipes escolhidos e apaga o tar.
       Variavel de ambiente HF_TOKEN e usada se o repo exigir login. Retomavel.
  cv   Common Voice: --root = pasta com validated.tsv + clips/ (colunas client_id, path, sentence;
       se existirem up_votes/down_votes exige up>=2 e down==0). Locutor = client_id (hash anonimo).
  mls  Multilingual LibriSpeech: --root = pasta com train|dev|test/{transcripts.txt,audio/<spk>/<book>/*.flac|opus}.
       ATENCAO: segmentos de 10-20 s (> limite de 10,2 s do treino) e transcricao SEM pontuacao/caixa.
       Sem --allow-truncate quase tudo e descartado; com ele o texto e cortado proporcionalmente
       (aproximado!) — opt-in consciente.
  tsv  --manifest arquivo TSV com colunas audio,text,speaker (audio relativo a --root ou absoluto).

Filtros (barato, antes do codec): minimo de clipes por voz, teto por voz, duracao 2,5-10,2 s apos trim
e SNR estimado >= --min-snr-db (o ALVO e supervisionado: audio ruidoso ensinaria o modelo a gerar ruido).
LICENCAS: CV = CC0; MLS = CC BY 4.0 (atribuicao). Confira os termos da fonte que baixou.

Uso:  python ptbr_lora\\data\\ingest_extra_corpus.py --format cv --root D:\\cv-pt --corpus cv_pt
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MIN_S, MAX_S = 2.5, 10.2


def _h(seed: int, key: str) -> str:
    return hashlib.sha1(f"{seed}|{key}".encode()).hexdigest()


def _clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()


# ------------------------------------------------------------------ parsers (puros)
def normalize_gender(g) -> str | None:
    """'F' | 'M' | None. CV novo: male_masculine/female_feminine; antigo: male/female."""
    g = str(g or "").strip().lower()
    if g.startswith("female") or g == "f":
        return "F"
    if g.startswith("male") or g == "m":
        return "M"
    return None


def parse_cv_hf(root: Path, corpus: str = "cv_pt", lang: str = "pt",
                splits=("train", "dev", "test")) -> list[dict]:
    """Le root/transcript/<lang>/<split>.tsv (layout do repo HF). `src` = root/clips/<arquivo>.mp3."""
    root = Path(root)
    rows, seen = [], set()
    for sp in splits:
        tsv = root / "transcript" / lang / f"{sp}.tsv"
        if not tsv.is_file():
            continue
        with tsv.open("r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                up, down = r.get("up_votes"), r.get("down_votes")
                if up not in (None, "") and down not in (None, ""):
                    if int(up) < 2 or int(down) > 0:
                        continue
                text = _clean_text(r.get("sentence") or r.get("text") or "")
                cid, path = r.get("client_id"), r.get("path")
                if not (cid and path and text):
                    continue
                fname = os.path.basename(path)
                if not fname.endswith(".mp3"):
                    fname += ".mp3"
                if fname in seen:
                    continue
                seen.add(fname)
                rows.append({"idx": f"{corpus}_{fname[:-4]}", "src": str(root / "clips" / fname), "fname": fname,
                             "text": text, "speaker": f"{corpus}:{cid[:16]}", "corpus": corpus,
                             "gender": normalize_gender(r.get("gender")), "split": sp})
    return rows


def select_cv_candidates(rows: list[dict], min_per: int, max_per: int, max_voices_per_gender: int,
                         seed: int = 42) -> tuple[list[dict], dict]:
    """Vozes com genero F/M e >= min_per clipes; ate `max_voices_per_gender` por genero (por hash) e
    `max_per` clipes por voz. O balanceamento FINAL M=F e feito em voices.balance_gender (depois do
    filtro de qualidade); aqui so se limita o que sera baixado/codificado."""
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("gender"):
            by[r["speaker"]].append(r)
    elig = {"F": [], "M": []}
    for spk, lst in by.items():
        if len(lst) >= min_per:
            elig[lst[0]["gender"]].append(spk)
    stats = {"eligible_F": len(elig["F"]), "eligible_M": len(elig["M"]),
             "speakers_total": len({r["speaker"] for r in rows}),
             "unknown_gender_speakers": len({r["speaker"] for r in rows if not r.get("gender")})}
    out = []
    for g in ("F", "M"):
        keep = sorted(elig[g], key=lambda k: _h(seed, f"v|{k}"))[:max_voices_per_gender]
        stats[f"selected_{g}"] = len(keep)
        for spk in keep:
            out.extend(sorted(by[spk], key=lambda r: _h(seed, r["idx"]))[:max_per])
    return out, stats


def parse_cv(root: Path, corpus: str = "cv_pt") -> list[dict]:
    root = Path(root)
    tsv = root / "validated.tsv"
    rows = []
    with tsv.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            up, down = r.get("up_votes"), r.get("down_votes")
            if up not in (None, "") and down not in (None, ""):
                if int(up) < 2 or int(down) > 0:
                    continue
            text = _clean_text(r.get("sentence") or r.get("text") or "")
            cid, path = r.get("client_id"), r.get("path")
            if not (cid and path and text):
                continue
            stem = Path(path).stem
            rows.append({"idx": f"{corpus}_{stem}", "src": str(root / "clips" / path), "text": text,
                         "speaker": f"{corpus}:{cid[:16]}", "corpus": corpus,
                         "gender": normalize_gender(r.get("gender"))})
    return rows


def parse_mls(root: Path, corpus: str = "mls_pt", splits=("train", "dev", "test")) -> list[dict]:
    root = Path(root)
    rows = []
    for sp in splits:
        tr = root / sp / "transcripts.txt"
        if not tr.is_file():
            continue
        for ln in tr.read_text(encoding="utf-8").splitlines():
            if "\t" not in ln:
                continue
            uid, text = ln.split("\t", 1)
            parts = uid.split("_")
            if len(parts) < 3 or not _clean_text(text):
                continue
            spk, book = parts[0], parts[1]
            base = root / sp / "audio" / spk / book / uid
            src = next((str(base.with_suffix(e)) for e in (".flac", ".opus") if base.with_suffix(e).exists()),
                       str(base.with_suffix(".flac")))
            rows.append({"idx": f"{corpus}_{uid}", "src": src, "text": _clean_text(text),
                         "speaker": f"{corpus}:{spk}", "corpus": corpus})
    return rows


def parse_tsv(manifest: Path, root: Path | None, corpus: str) -> list[dict]:
    rows = []
    with Path(manifest).open("r", encoding="utf-8", newline="") as f:
        for i, r in enumerate(csv.DictReader(f, delimiter="\t")):
            a = Path(r["audio"])
            src = a if a.is_absolute() or root is None else Path(root) / a
            text, spk = _clean_text(r.get("text", "")), (r.get("speaker") or "").strip()
            if not (text and spk):
                continue
            rows.append({"idx": f"{corpus}_{a.stem}_{i}", "src": str(src), "text": text,
                         "speaker": f"{corpus}:{spk}", "corpus": corpus})
    return rows


def limit_per_speaker(rows: list[dict], min_per: int, max_per: int, seed: int = 42) -> list[dict]:
    """Corta cada voz em `max_per` clipes (amostra deterministica por hash); remove vozes com < min_per."""
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["speaker"]].append(r)
    out = []
    for spk in sorted(by):
        lst = by[spk]
        if len(lst) < min_per:
            continue
        out.extend(sorted(lst, key=lambda r: _h(seed, r["idx"]))[:max_per])
    return out


# ------------------------------------------------------------------ qualidade (numpy)
def estimate_snr_db(wav: np.ndarray, sr: int, frame_s: float = 0.03) -> float:
    """SNR grosseiro: energia dos 90 % mais fortes dos quadros vs. dos 10 % mais fracos (piso de ruido)."""
    n = max(1, int(sr * frame_s))
    m = len(wav) // n
    if m < 8:
        return float("nan")
    e = np.mean(wav[: m * n].reshape(m, n) ** 2, axis=1) + 1e-12
    e = np.sort(e)
    noise = float(np.mean(e[: max(1, m // 10)]))
    sig = float(np.mean(e[-max(1, m // 3):]))
    return 10.0 * float(np.log10(sig / noise))


def quality_ok(wav: np.ndarray, sr: int, min_snr_db: float) -> bool:
    if min_snr_db <= 0:
        return True
    s = estimate_snr_db(wav, sr)
    return bool(s == s and s >= min_snr_db)


# ------------------------------------------------------------------ download (stdlib, retomavel)
HF_REPO = "fsicoli/common_voice_19_0"


def hf_url(repo: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def http_download(url: str, dest: Path, token: str | None = None, tries: int = 6) -> bool:
    """Baixa `url` em `dest` (retoma de dest.part). False se 404."""
    dest = Path(dest)
    if dest.exists():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(tries):
        have = part.stat().st_size if part.exists() else 0
        hdr = {"User-Agent": "ptbr-lora-ingest"}
        if token:
            hdr["Authorization"] = f"Bearer {token}"
        if have:
            hdr["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=60) as resp:
                mode = "ab" if (have and resp.status == 206) else "wb"
                done = have if mode == "ab" else 0
                t0, last = time.time(), 0
                with part.open(mode) as f:
                    while True:
                        buf = resp.read(1 << 20)
                        if not buf:
                            break
                        f.write(buf)
                        done += len(buf)
                        if done - last >= 200 * (1 << 20):
                            last = done
                            print(f"[dl] {dest.name}: {done / 2**30:.2f} GB "
                                  f"({done / max(time.time() - t0, 1e-9) / 2**20:.1f} MB/s)", flush=True)
            part.replace(dest)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            if e.code == 416:                                   # ja completo
                part.replace(dest)
                return True
            if e.code in (401, 403):
                sys.exit(f"[dl] HTTP {e.code} em {url}\n     o repo pode exigir login: aceite os termos no site e "
                         f"defina HF_TOKEN (set HF_TOKEN=hf_...) antes de rodar.")
            print(f"[dl] HTTP {e.code}; tentativa {attempt + 1}/{tries}", flush=True)
        except Exception as e:  # noqa: BLE001  (rede instavel: retoma)
            print(f"[dl] {type(e).__name__}: {str(e)[:80]}; tentativa {attempt + 1}/{tries}", flush=True)
        time.sleep(3 * (attempt + 1))
    sys.exit(f"[dl] falhou apos {tries} tentativas: {url}")


def extract_wanted(tar_path: Path, wanted: set, clips_dir: Path) -> int:
    """Extrai de `tar_path` so os arquivos cujo NOME (basename) esta em `wanted`. Retorna quantos."""
    import tarfile

    clips_dir = Path(clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(tar_path) as tf:
        for m in tf:
            if not m.isfile():
                continue
            base = os.path.basename(m.name)
            dst = clips_dir / base
            if base in wanted and not dst.exists():
                src = tf.extractfile(m)
                if src is None:
                    continue
                with dst.open("wb") as f:
                    f.write(src.read())
                n += 1
    return n


def fetch_cv_audio(rows: list[dict], root: Path, repo: str, lang: str, splits, token, keep_tars: bool) -> int:
    """Baixa os tars de audio de cada split e extrai so os clipes de `rows`. Retomavel (marcadores .done)."""
    root = Path(root)
    wanted = {r["fname"] for r in rows}
    ns_path = root / "n_shards.json"
    http_download(hf_url(repo, "n_shards.json"), ns_path, token)
    try:
        ns = json.loads(ns_path.read_text(encoding="utf-8"))[lang]
    except Exception:  # noqa: BLE001
        ns = {}
    got = 0
    for sp in splits:
        want_sp = {r["fname"] for r in rows if r.get("split") == sp}
        if not want_sp:
            continue
        n_sh = ns.get(sp)
        k = 0
        while n_sh is None or k < int(n_sh):
            name = f"{lang}_{sp}_{k}.tar"
            mark = root / "shards_done" / (name + ".done")
            tar = root / "tars" / name
            if mark.exists():
                k += 1
                continue
            ok = http_download(hf_url(repo, f"audio/{lang}/{sp}/{name}"), tar, token)
            if not ok:
                break                                             # fim dos shards (n_shards desconhecido)
            n = extract_wanted(tar, wanted, root / "clips")
            got += n
            mark.parent.mkdir(parents=True, exist_ok=True)
            mark.write_text(str(n), encoding="utf-8")
            if not keep_tars:
                tar.unlink(missing_ok=True)
            print(f"[cv] {name}: {n} clipes extraidos (total {got}/{len(wanted)})", flush=True)
            k += 1
    return got


# ------------------------------------------------------------------ processamento (GPU)
def process(rows: list[dict], out: Path, device: str, min_snr_db: float, allow_truncate: bool) -> None:
    import librosa
    import soundfile as sf

    _core = Path(__file__).resolve().parents[1] / "core"
    sys.path.insert(0, str(_core))
    sys.path.insert(0, str(_core.parents[1]))
    import common_breeze as CB
    from prepare_dataset import truncate_pair

    Qwen3TTSTokenizer = CB.import_qwen_tts()
    atok = Qwen3TTSTokenizer.from_pretrained(str(CB.CKPT / "audio_tokenizer"), device_map=device)
    (out / "tokens").mkdir(parents=True, exist_ok=True)
    (out / "wavs24").mkdir(parents=True, exist_ok=True)
    meta_p = out / "dataset_meta.jsonl"
    done = set()
    if meta_p.exists():
        done = {json.loads(ln)["idx"] for ln in meta_p.read_text(encoding="utf-8").splitlines() if ln.strip()}
    todo = [r for r in rows if r["idx"] not in done]
    print(f"[ingest] total={len(rows)} ja_feitos={len(done)} a_processar={len(todo)}", flush=True)
    meta_f = meta_p.open("a", encoding="utf-8")
    stats, t0 = Counter(), time.time()
    for k, row in enumerate(todo):
        try:
            try:
                wav, sr = sf.read(row["src"], dtype="float32", always_2d=True)
                wav = wav.mean(axis=1)
            except Exception:  # noqa: BLE001  (mp3/opus sem suporte do libsndfile)
                wav, sr = librosa.load(row["src"], sr=None, mono=True)
            dur = len(wav) / sr
            text, truncated = row["text"], False
            if dur > MAX_S:
                if not allow_truncate:
                    stats["drop_long"] += 1
                    continue
                wav, text = truncate_pair(wav, sr, text, MAX_S)
                truncated = True
                if len(text) < 20:
                    stats["drop_trunc_text"] += 1
                    continue
            wav, _ = librosa.effects.trim(wav, top_db=40)
            d = len(wav) / sr
            if not (MIN_S <= d <= MAX_S + 0.3):
                stats["drop_dur"] += 1
                continue
            if not quality_ok(wav, sr, min_snr_db):
                stats["drop_snr"] += 1
                continue
            w24 = librosa.resample(wav, orig_sr=sr, target_sr=CB.SR)
            peak = float(np.max(np.abs(w24))) if len(w24) else 0.0
            if peak < 0.30 or peak > 0.99:
                w24 = w24 * (CB.PEAK_NORM / max(peak, 1e-9))
            w24 = w24.clip(-1.0, 1.0).astype("float32")
            codes = atok.encode(w24, sr=CB.SR)["audio_codes"][0]
            codes = codes.detach().cpu().numpy().astype("int16") if hasattr(codes, "cpu") else codes
            assert codes.ndim == 2 and codes.shape[1] == 16 and 0 <= codes.min() and codes.max() < 2051
            np.savez_compressed(out / "tokens" / f"{row['idx']}.npz", codes=codes)
            sf.write(out / "wavs24" / f"{row['idx']}.wav", w24, CB.SR, subtype="PCM_16")
            rec = {"idx": row["idx"], "wav_rel": f"wavs/{row['idx']}.wav", "corpus": row["corpus"],
                   "speaker": row["speaker"], "text": text, "dur_src_s": round(dur, 3),
                   "dur_proc_s": round(len(w24) / CB.SR, 3), "frames": int(codes.shape[0]),
                   "truncated": truncated}
            if row.get("gender"):
                rec["gender"] = row["gender"]
            meta_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            meta_f.flush()
            stats["ok_trunc" if truncated else "ok"] += 1
        except Exception as exc:  # noqa: BLE001
            stats["error"] += 1
            print(f"[ingest] ERRO {row['idx']}: {type(exc).__name__}: {str(exc)[:100]}", flush=True)
        if (k + 1) % 200 == 0:
            rate = (time.time() - t0) / (k + 1)
            print(f"[ingest] {k + 1}/{len(todo)} {dict(stats)} eta={rate * (len(todo) - k - 1) / 60:.1f}min",
                  flush=True)
    meta_f.close()
    print(f"[ingest] FIM {dict(stats)} em {(time.time() - t0) / 60:.1f} min -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--format", choices=["cvhf", "cv", "mls", "tsv"], required=True)
    ap.add_argument("--root", default=None, help="cvhf: pasta de trabalho (default <ARTIFACTS>/cv_raw)")
    ap.add_argument("--hf-repo", default=HF_REPO)
    ap.add_argument("--lang", default="pt")
    ap.add_argument("--splits", nargs="*", default=["train", "dev", "test"])
    ap.add_argument("--max-voices-per-gender", type=int, default=150)
    ap.add_argument("--keep-tars", action="store_true")
    ap.add_argument("--manifest", default=None, help="so p/ --format tsv")
    ap.add_argument("--corpus", default=None, help="nome do corpus (default cv_pt / mls_pt / extra)")
    ap.add_argument("--out", default=None, help="pasta de saida (default <ARTIFACTS>/extra_<corpus>)")
    ap.add_argument("--min-per-speaker", type=int, default=12)
    ap.add_argument("--max-per-speaker", type=int, default=30)
    ap.add_argument("--min-snr-db", type=float, default=15.0, help="0 = sem filtro (estimador grosseiro)")
    ap.add_argument("--allow-truncate", action="store_true", help="MLS: corta audio/texto > 10,2 s (aproximado)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true", help="so parseia e mostra as contagens (sem GPU)")
    args = ap.parse_args()

    corpus = args.corpus or {"cvhf": "cv_pt", "cv": "cv_pt", "mls": "mls_pt", "tsv": "extra"}[args.format]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
    import paths

    out = Path(args.out) if args.out else paths.ARTIFACTS / f"extra_{corpus}"
    if args.format == "cvhf":
        root = Path(args.root) if args.root else paths.ARTIFACTS / "cv_raw"
        token = os.environ.get("HF_TOKEN")
        for sp in args.splits:                                         # so os TSVs (poucos MB)
            http_download(hf_url(args.hf_repo, f"transcript/{args.lang}/{sp}.tsv"),
                          root / "transcript" / args.lang / f"{sp}.tsv", token)
        rows = parse_cv_hf(root, corpus, args.lang, tuple(args.splits))
        rows, st = select_cv_candidates(rows, args.min_per_speaker, args.max_per_speaker,
                                        args.max_voices_per_gender)
        print(f"[cv] vozes no manifesto: {st['speakers_total']} (genero desconhecido: "
              f"{st['unknown_gender_speakers']}) | elegiveis (>= {args.min_per_speaker} clipes): "
              f"F={st['eligible_F']} M={st['eligible_M']} | selecionadas: F={st['selected_F']} M={st['selected_M']} "
              f"| clipes={len(rows)}", flush=True)
        print("[cv] o balanceamento final F=M (menor dos dois) e aplicado em build_diverse_set.py", flush=True)
        if args.limit:
            rows = rows[: args.limit]
        if args.dry_run or not rows:
            return
        fetch_cv_audio(rows, root, args.hf_repo, args.lang, args.splits, token, args.keep_tars)
        rows = [r for r in rows if Path(r["src"]).exists()]
        print(f"[cv] clipes presentes em disco: {len(rows)}", flush=True)
        out.mkdir(parents=True, exist_ok=True)
        process(rows, out, args.device, args.min_snr_db, False)
        return
    root = Path(args.root)
    if args.format == "cv":
        rows = parse_cv(root, corpus)
    elif args.format == "mls":
        rows = parse_mls(root, corpus)
    else:
        assert args.manifest, "--format tsv exige --manifest"
        rows = parse_tsv(Path(args.manifest), root, corpus)
    n_spk0 = len({r["speaker"] for r in rows})
    rows = limit_per_speaker(rows, args.min_per_speaker, args.max_per_speaker)
    if args.limit:
        rows = rows[: args.limit]
    print(f"[ingest] {corpus}: {n_spk0} vozes no manifesto -> {len({r['speaker'] for r in rows})} vozes com "
          f">= {args.min_per_speaker} clipes | {len(rows)} clipes a processar (teto {args.max_per_speaker}/voz)")
    if args.dry_run or not rows:
        return
    out.mkdir(parents=True, exist_ok=True)
    process(rows, out, args.device, args.min_snr_db, args.allow_truncate)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
