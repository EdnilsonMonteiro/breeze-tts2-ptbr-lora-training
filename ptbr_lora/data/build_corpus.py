"""build_corpus.py — ingestao dos corpora novos (TAGARELA / CML-TTS PT / CETUC)
para o formato do projeto: datasets/<corpus>/wavs/<idx>.wav + texts.csv + speakers.jsonl.

Duas passadas (memoria baixa):
  1) scan: le metadados + duracao (sem decodificar tudo) e agrupa por grupo
     (show/cliente/locutor);
  2) selecao round-robin por grupo (diversidade, sem repeticao via manifest);
  3) materializa: decodifica e grava SOMENTE os clipes selecionados.

Manifest `selection.jsonl` por corpus (chave unica) -> resumivel e escalavel.

Uso:
  python scripts/build_corpus.py --corpus tagarela --hours 120
  python scripts/build_corpus.py --corpus cml_pt   --hours 35
  python scripts/build_corpus.py --corpus cetuc    --hours 30
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import tarfile
import unicodedata
from pathlib import Path

import num2words
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
import paths  # noqa: E402

RAW = paths.DATASETS_ROOT / "_raw"
DATASETS = paths.DATASETS_ROOT

MIN_DUR, MAX_DUR = 4.0, 10.2
CHARS_PER_S = (4.0, 30.0)
ALLOWED = re.compile(r"[^A-Za-zÀ-ÿ0-9 .,!?'’\-:;()\"]+")
NUM_RE = re.compile(r"(?<![\w])(\d{1,4})(?![\w])")
PREFIX = {"tagarela": "tag", "cml_pt": "cml", "cetuc": "cetuc"}


def clean_text(t: str, dur: float, corpus: str) -> str | None:
    t = unicodedata.normalize("NFC", t or "")
    t = t.replace("…", "...").replace("–", "-").replace("—", "-")
    t = ALLOWED.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip(" -")
    if not t:
        return None
    if not (CHARS_PER_S[0] <= len(t) / max(dur, 1e-6) <= CHARS_PER_S[1]):
        return None
    if corpus != "cetuc":
        t = NUM_RE.sub(lambda m: num2words.num2words(int(m.group(1)), lang="pt-BR"), t)
    else:
        t = t[0].upper() + t[1:]
        if t[-1] not in ".!?":
            t += "."
    return t


def _dur_bytes(b: bytes) -> float:
    return float(sf.info(io.BytesIO(b)).duration)


def _decode(b: bytes) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
    return wav.mean(axis=1), sr


def _show_of(path: str) -> str:
    for p in str(path).split("/"):
        if p.startswith("show_"):
            return p
    return "unknown"


# ============================================================ TAGARELA
def scan_tagarela(max_scan: int | None):
    import pyarrow.parquet as pq

    files = sorted((RAW / "tagarela" / "default" / "tts").glob("*.parquet"))
    if not files:
        sys.exit("[ingest] nenhum parquet em datasets/_raw/tagarela/default/tts")
    groups: dict[str, list[dict]] = {}
    gi = 0
    for fp in files:
        row = 0
        for batch in pq.ParquetFile(fp).iter_batches(
                batch_size=128, columns=["audio", "sentence", "path"]):
            cols = batch.to_pydict()
            for a, sent, path in zip(cols["audio"], cols["sentence"], cols["path"]):
                loc = (str(fp), row); row += 1; gi += 1
                if max_scan and gi > max_scan:
                    return groups
                if not a or not a.get("bytes"):
                    continue
                try:
                    dur = _dur_bytes(a["bytes"])
                except Exception:  # noqa: BLE001
                    continue
                if not (MIN_DUR <= dur <= MAX_DUR):
                    continue
                txt = clean_text(sent, dur, "tagarela")
                if not txt:
                    continue
                g = _show_of(path)
                groups.setdefault(g, []).append(
                    {"key": str(path), "group": g, "dur": dur, "text": txt, "loc": loc})
    return groups


def materialize_tagarela(selected: list[dict], writer):
    import pyarrow.parquet as pq

    byfile: dict[str, dict[int, dict]] = {}
    for m in selected:
        byfile.setdefault(m["loc"][0], {})[m["loc"][1]] = m
    for fp, idxmap in byfile.items():
        row = 0
        for batch in pq.ParquetFile(fp).iter_batches(batch_size=128, columns=["audio"]):
            for a in batch.column("audio").to_pylist():
                m = idxmap.get(row)
                if m is not None and a and a.get("bytes"):
                    writer(m, a["bytes"])
                row += 1


# ============================================================ CML-TTS PT
def scan_cml(max_scan: int | None):
    import pyarrow.parquet as pq

    files = sorted((RAW / "cml_pt" / "data").glob("train-*.parquet"))
    if not files:
        sys.exit("[ingest] nenhum parquet em datasets/_raw/cml_pt/data")
    groups: dict[str, list[dict]] = {}
    gi = 0
    for fp in files:
        row = 0
        for batch in pq.ParquetFile(fp).iter_batches(
                batch_size=256,
                columns=["transcript", "client_id", "duration", "filename"]):
            cols = batch.to_pydict()
            for txt, cid, dur, fn in zip(cols["transcript"], cols["client_id"],
                                         cols["duration"], cols["filename"]):
                loc = (str(fp), row); row += 1; gi += 1
                if max_scan and gi > max_scan:
                    return groups
                if dur is None or not (MIN_DUR <= float(dur) <= MAX_DUR):
                    continue
                t = clean_text(txt, float(dur), "cml_pt")
                if not t:
                    continue
                g = f"cml:{cid}"
                groups.setdefault(g, []).append(
                    {"key": str(fn), "group": g, "dur": float(dur), "text": t, "loc": loc})
    return groups


def materialize_cml(selected: list[dict], writer):
    import pyarrow.parquet as pq

    byfile: dict[str, dict[int, dict]] = {}
    for m in selected:
        byfile.setdefault(m["loc"][0], {})[m["loc"][1]] = m
    for fp, idxmap in byfile.items():
        row = 0
        for batch in pq.ParquetFile(fp).iter_batches(batch_size=128, columns=["audio"]):
            for a in batch.column("audio").to_pylist():
                m = idxmap.get(row)
                if m is not None and a and a.get("bytes"):
                    writer(m, a["bytes"])
                row += 1


# ============================================================ CETUC
def scan_cetuc(max_scan: int | None):
    tars = sorted((RAW / "cetuc" / "data" / "train").rglob("*.tar.gz"))
    if not tars:
        sys.exit("[ingest] nenhum tar.gz em datasets/_raw/cetuc/data/train")
    groups: dict[str, list[dict]] = {}
    gi = 0
    for tp in tars:
        speaker = tp.parent.name
        with tarfile.open(tp, "r|gz") as tf:  # streaming: le uma vez, sequencial
            pending: dict[str, tuple[str, bytes]] = {}
            for m in tf:
                if not m.isfile():
                    continue
                nm = m.name
                if nm.startswith("._") or "/._" in nm:
                    continue
                if nm.endswith(".txt") or nm.endswith(".json"):
                    ext = nm.rsplit(".", 1)[1]
                    pending[nm[: -(len(ext) + 1)]] = (ext, tf.extractfile(m).read())
                elif nm.endswith(".wav"):
                    item = pending.pop(nm[:-4], None)
                    if item is None:
                        continue
                    gi += 1
                    if max_scan and gi > max_scan:
                        return groups
                    wav_b = tf.extractfile(m).read()
                    try:
                        dur = _dur_bytes(wav_b)
                    except Exception:  # noqa: BLE001
                        continue
                    if not (MIN_DUR <= dur <= MAX_DUR):
                        continue
                    ext, data = item
                    raw_txt = data.decode("utf-8", "replace")
                    if ext == "json":
                        try:
                            raw_txt = json.loads(raw_txt).get("text", raw_txt)
                        except Exception:  # noqa: BLE001
                            pass
                    t = clean_text(raw_txt, dur, "cetuc")
                    if not t:
                        continue
                    g = f"cetuc:{speaker}"
                    groups.setdefault(g, []).append(
                        {"key": f"{speaker}/{nm[:-4]}", "group": g, "dur": dur, "text": t,
                         "loc": (str(tp), nm)})
    return groups


def materialize_cetuc(selected: list[dict], writer):
    bytar: dict[str, dict[str, dict]] = {}
    for m in selected:
        bytar.setdefault(m["loc"][0], {})[m["loc"][1]] = m
    for tp, namemap in bytar.items():
        with tarfile.open(tp, "r|gz") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                hit = namemap.get(m.name)
                if hit is not None:
                    f = tf.extractfile(m)
                    if f is not None:
                        writer(hit, f.read())


# ============================================================ driver
CORPORA = {
    "tagarela": (scan_tagarela, materialize_tagarela),
    "cml_pt": (scan_cml, materialize_cml),
    "cetuc": (scan_cetuc, materialize_cetuc),
}


def build(corpus: str, hours: float, max_scan: int | None) -> None:
    out_dir = DATASETS / corpus
    wavs_dir = out_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    manifest_p = out_dir / "selection.jsonl"
    done: set[str] = set()
    start_n = 0
    pref = PREFIX[corpus]
    if manifest_p.exists():
        for ln in manifest_p.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            done.add(r["key"])
            i = str(r.get("idx", ""))
            if i.startswith(pref + "-"):
                try:
                    start_n = max(start_n, int(i.split("-")[-1]) + 1)
                except ValueError:
                    pass
        print(f"[ingest] {corpus}: {len(done)} itens no manifest | proximo idx={start_n}",
              flush=True)

    scan, materialize = CORPORA[corpus]
    print(f"[ingest] {corpus}: scan...", flush=True)
    groups = scan(max_scan)
    for g in groups:
        groups[g] = [c for c in groups[g] if c["key"] not in done]
        groups[g].sort(key=lambda c: hashlib.sha1(c["key"].encode()).hexdigest())
    groups = {g: v for g, v in groups.items() if v}
    n_valid = sum(len(v) for v in groups.values())
    print(f"[ingest] {corpus}: {len(groups)} grupos, {n_valid} clipes validos", flush=True)

    target_s = hours * 3600
    order = sorted(groups)
    ptr = {g: 0 for g in order}
    selected: list[dict] = []
    total_s = 0.0
    exhausted = False
    while total_s < target_s and not exhausted:
        exhausted = True
        for g in order:
            if total_s >= target_s:
                break
            lst = groups[g]
            if ptr[g] >= len(lst):
                continue
            exhausted = False
            c = lst[ptr[g]]; ptr[g] += 1
            selected.append(c)
            total_s += c["dur"]
    print(f"[ingest] {corpus}: selecionados {len(selected)} clipes / "
          f"{total_s/3600:.2f} h de {len(order)} grupos", flush=True)

    pref = PREFIX[corpus]
    texts_f = (out_dir / "texts.csv").open("a", encoding="utf-8")
    spk_f = (out_dir / "speakers.jsonl").open("a", encoding="utf-8")
    man_f = manifest_p.open("a", encoding="utf-8")
    n_written = start_n

    def writer(m: dict, wav_b: bytes) -> None:
        nonlocal n_written
        try:
            wav, sr = _decode(wav_b)
        except Exception:  # noqa: BLE001
            return
        idx = f"{pref}-{n_written:06d}"
        sf.write(str(wavs_dir / f"{idx}.wav"), np.clip(wav, -1, 1), sr, subtype="PCM_16")
        texts_f.write(f"wavs/{idx}.wav=={m['text']}\n")
        spk_f.write(json.dumps({"idx": idx, "speaker": m["group"]}, ensure_ascii=False) + "\n")
        man_f.write(json.dumps({"idx": idx, "key": m["key"], "group": m["group"],
                                "dur": round(m["dur"], 3)}, ensure_ascii=False) + "\n")
        n_written += 1
        if n_written % 1000 == 0:
            texts_f.flush(); spk_f.flush(); man_f.flush()
            print(f"[ingest] {corpus}: {n_written}/{len(selected)} gravados", flush=True)

    materialize(selected, writer)
    texts_f.close(); spk_f.close(); man_f.close()
    print(f"[ingest] {corpus} FIM: {n_written} clipes gravados", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=list(CORPORA))
    ap.add_argument("--hours", type=float, required=True)
    ap.add_argument("--max-scan", type=int, default=None)
    args = ap.parse_args()
    build(args.corpus, args.hours, args.max_scan)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
