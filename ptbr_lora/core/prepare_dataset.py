"""prepare_dataset.py — FASE B: preparacao do TTS-Portuguese-Corpus p/ LoRA PT-BR.

Subcomandos:
  process   [--limit N] [--device cuda]     parse+clean+resample+encode -> tokens/,wavs24/,meta
  finalize  [--gold N] [--parity-device cuda]  splits 90/5/5 + gold dumps + paridade oficial

Resumivel: amostras com tokens/<idx>.npz existentes sao puladas no 'process'.
Uso: C:/IA/Breeze-tts/breeze-tts/venv/Scripts/python.exe prepare_dataset.py process --limit 40
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_CORE = Path(__file__).resolve().parent
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))

import common_breeze as CB
CB.TRAINING.mkdir(parents=True, exist_ok=True)

META_JSONL = CB.TRAINING / "dataset_meta.jsonl"
MANIFEST_CSV = CB.TRAINING / "manifest.csv"
SUMMARY = CB.TRAINING / "summary.json"

LINE_RE = re.compile(r"^(wavs/[^\s]+\.wav)==(.*)$")


# ------------------------------------------------------------------ csv


def _speaker_map_for(corp: dict) -> dict[str, str]:
    """idx -> speaker. Prioridade: speakers.jsonl do corpus; podcast legado usa
    chunks_index.jsonl; tata fica sem mapa (default 'tata')."""
    name = corp["name"]
    root = CB.corpus_dir(corp)
    m: dict[str, str] = {}
    spk_file = root / "speakers.jsonl"
    if spk_file.exists():
        for ln in spk_file.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                m[r["idx"]] = r["speaker"]
        return m
    if name == "podcast":
        ci = CB.SCRAPING_WORK / "chunks_index.jsonl"
        if ci.exists():
            for ln in ci.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    r = json.loads(ln)
                    m[r["chunk"].removesuffix(".wav")] = f"{r['tag']}:{r['speaker']}"
    return m


def parse_csv() -> list[dict]:
    rows: list[dict] = []
    seen = Counter()
    n_bad = n_dupe = 0
    for corp in CB.load_corpora():
        name = corp["name"]
        root = CB.corpus_dir(corp)
        csv = CB.corpus_csv(corp)
        if not csv.exists():
            print(f"[parse] corpus '{name}': csv ausente ({csv}) -> pulando")
            continue
        spk_map = _speaker_map_for(corp)
        raw = csv.read_text(encoding="utf-8")
        n_c = 0
        for ln_no, line in enumerate(raw.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            m = LINE_RE.match(line)
            if not m:
                n_bad += 1
                print(f"[parse] {name} linha invalida #{ln_no}: {line[:60]!r}")
                continue
            rel, text = m.group(1), m.group(2)
            key = f"{name}:{rel}"
            seen[key] += 1
            if seen[key] > 1:
                n_dupe += 1
                continue
            text = html.unescape(text)
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                n_bad += 1
                continue
            idx = Path(rel).stem
            speaker = spk_map.get(idx, "tata" if name == "tata" else idx)
            rows.append({"wav_rel": rel, "idx": idx, "text": text,
                         "corpus": name, "speaker": speaker, "root": str(root)})
            n_c += 1
        missing = [r for r in rows if r["corpus"] == name
                   and not (root / r["wav_rel"]).exists()]
        if missing:
            print(f"[parse] {name}: AVISO {len(missing)} wavs ausentes "
                  f"(ex.: {[r['wav_rel'] for r in missing[:3]]})")
        print(f"[parse] {name}: validos={n_c} ausentes={len(missing)}")
    print(f"[parse] TOTAL validos={len(rows)} dupes={n_dupe} invalidas/vazias={n_bad}")
    miss_keys = {(r["corpus"], r["wav_rel"]) for r in rows
                 if not (Path(r["root"]) / r["wav_rel"]).exists()}
    return [r for r in rows if (r["corpus"], r["wav_rel"]) not in miss_keys]


# ------------------------------------------------------------------ process


def process(limit: int | None, device: str) -> None:
    import librosa
    import soundfile as sf
    from qwen_tts import Qwen3TTSTokenizer

    rows = parse_csv()
    n_done_pre = sum(1 for r in rows if (CB.TOKENS_DIR / f"{r['idx']}.npz").exists())
    todo = [r for r in rows if not (CB.TOKENS_DIR / f"{r['idx']}.npz").exists()]
    if limit:
        todo = todo[:limit]
    print(f"[process] total_csv={len(rows)} feitos={n_done_pre} a_processar={len(todo)}")

    CB.TOKENS_DIR.mkdir(parents=True, exist_ok=True)
    CB.WAVS24_DIR.mkdir(parents=True, exist_ok=True)
    meta_f = META_JSONL.open("a", encoding="utf-8")
    t_start = time.time()
    stats = Counter()
    atok = Qwen3TTSTokenizer.from_pretrained(str(CB.CKPT / "audio_tokenizer"), device_map=device)

    for k, row in enumerate(todo):
        src = Path(row["root"]) / row["wav_rel"]
        try:
            info = sf.info(src)
            dur = info.frames / info.samplerate
            if dur < CB.MIN_DUR_S:
                stats["drop_short"] += 1
                continue
            wav48, sr48 = sf.read(src, dtype="float32", always_2d=True)
            wav48 = wav48.mean(axis=1)
            text_used = row["text"]
            truncated = False
            if dur > CB.MAX_DUR_S:
                wav48, text_used = truncate_pair(wav48, sr48, row["text"], CB.MAX_DUR_S)
                truncated = True
                if len(text_used) < 20:
                    stats["drop_trunc_text"] += 1
                    continue
            wav48, _ = librosa.effects.trim(wav48, top_db=40)
            d_trim = len(wav48) / sr48
            if d_trim < CB.MIN_DUR_S:
                stats["drop_short_trim"] += 1
                continue
            if d_trim > CB.MAX_DUR_S + 0.3:
                stats["drop_long_trim"] += 1
                continue
            wav24 = librosa.resample(wav48, orig_sr=sr48, target_sr=CB.SR)
            peak = float(np.max(np.abs(wav24))) if len(wav24) else 0.0
            if peak < 0.30 or peak > 0.99:
                wav24 = wav24 * (CB.PEAK_NORM / max(peak, 1e-9))
            wav24 = wav24.clip(-1.0, 1.0).astype("float32")

            enc = atok.encode(wav24, sr=CB.SR)
            codes = enc["audio_codes"][0]
            codes = codes.detach().cpu().numpy().astype("int16") if hasattr(codes, "cpu") else codes
            assert codes.ndim == 2 and codes.shape[1] == 16, codes.shape
            assert codes.min() >= 0 and codes.max() < 2051, (codes.min(), codes.max())

            np.savez_compressed(CB.TOKENS_DIR / f"{row['idx']}.npz", codes=codes)
            sf.write(CB.WAVS24_DIR / f"{row['idx']}.wav", wav24, CB.SR, subtype="PCM_16")
            rec = {
                "idx": row["idx"],
                "wav_rel": row["wav_rel"],
                "corpus": row["corpus"],
                "speaker": row["speaker"],
                "text": text_used,
                "dur_src_s": round(dur, 3),
                "dur_proc_s": round(len(wav24) / CB.SR, 3),
                "frames": int(codes.shape[0]),
                "truncated": truncated,
            }
            meta_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            meta_f.flush()
            stats["ok_trunc" if truncated else "ok"] += 1
        except Exception as exc:  # noqa: BLE001
            stats["error"] += 1
            print(f"[process] ERRO {row['idx']}: {type(exc).__name__}: {exc}")
        if (k + 1) % 100 == 0:
            rate = (time.time() - t_start) / (k + 1)
            eta = rate * (len(todo) - k - 1) / 60
            print(f"[process] {k+1}/{len(todo)} ok={stats['ok']} eta={eta:.1f}min")

    meta_f.close()
    print(f"[process] FIM {dict(stats)} em {(time.time()-t_start)/60:.1f} min")


def np_absmax(a):
    return float(np.max(np.abs(a))) if len(a) else 0.0


PUNCT_CUT_RE = re.compile(r"[.!?;:](\s|$)")


def truncate_pair(wav48, sr, text: str, cap_s: float):
    """Trunca audio no vale de silencio mais proximo do limite e corta o texto
    proporcionalmente por palavras (corpus de leitura = ritmo uniforme).
    Retorna (wav_truncado, texto_ajustado). Chamador valida limites minimos."""
    import librosa

    intervals = librosa.effects.split(wav48, top_db=40)
    cap_n = int(cap_s * sr)
    ends = [int(e) for s, e in intervals if e <= cap_n]
    cut = ends[-1] if ends else cap_n
    wav = wav48[:cut]
    dur_total = max(len(wav48) / sr, 1e-9)
    frac = min(1.0, (len(wav) / sr) / dur_total)
    words = text.split()
    kw = max(4, int(round(len(words) * frac)))
    seg = " ".join(words[:kw])
    m = list(PUNCT_CUT_RE.finditer(seg + " "))
    if m:
        seg = seg[: m[-1].start()].strip()
    elif len(m) == 0 and "," in seg and kw < len(words):
        seg = seg[: seg.rfind(",")].strip()
    return wav, seg


# ------------------------------------------------------------------ finalize


def _apply_corpus_overrides(by_idx: dict[str, dict]) -> None:
    """Aplica speakers.jsonl (locutor) e ref_map.jsonl (referencia por similaridade)
    de cada corpus sobre os registros do meta."""
    for corp in CB.load_corpora():
        root = CB.corpus_dir(corp)
        name = corp["name"]
        spk_file = root / "speakers.jsonl"
        if spk_file.exists():
            for ln in spk_file.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                r = json.loads(ln)
                t = by_idx.get(r["idx"])
                if t is not None:
                    t["speaker"] = r["speaker"]
                    t.setdefault("corpus", name)
        ref_file = root / "ref_map.jsonl"
        if ref_file.exists():
            for ln in ref_file.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                r = json.loads(ln)
                t = by_idx.get(r["idx"])
                if t is not None and r.get("ref_idx"):
                    t["ref_idx"] = r["ref_idx"]


def load_meta() -> list[dict]:
    global _META_CACHE
    if _META_CACHE is not None:
        return _META_CACHE
    by_idx: dict[str, dict] = {}
    for ln in META_JSONL.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            by_idx[r["idx"]] = r  # ultima ocorrencia vence (idempotente)
    recs = list(by_idx.values())
    spk = _chunk_speaker_map()
    for r in recs:
        if "speaker" not in r:
            r["corpus"] = "podcast" if r["idx"] in spk else "tata"
            r["speaker"] = spk.get(r["idx"], "tata")
    _apply_corpus_overrides(by_idx)
    _META_CACHE = recs
    return recs


_META_CACHE: list[dict] | None = None


def invalidate_meta_cache() -> None:
    global _META_CACHE, _REC_BY_IDX
    _META_CACHE = None
    _REC_BY_IDX = None


def _chunk_speaker_map() -> dict[str, str]:
    """idx do podcast -> 'tag:SPEAKER_XX' (identidade NAO compartilha entre episodios)."""
    global _SPK_MAP
    if _SPK_MAP is not None:
        return _SPK_MAP
    m: dict[str, str] = {}
    ci = CB.SCRAPING_WORK / "chunks_index.jsonl"
    if ci.exists():
        for ln in ci.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                m[r["chunk"].removesuffix(".wav")] = f"{r['tag']}:{r['speaker']}"
    _SPK_MAP = m
    return m


_SPK_MAP: dict[str, str] | None = None


REF_EDIT_FRAC = 0.5


def set_ref_edit_frac(frac: float) -> None:
    """Fracao de exemplos em modo ref_edit (com referencia). 1.0 = todos."""
    global REF_EDIT_FRAC
    REF_EDIT_FRAC = min(1.0, max(0.0, float(frac)))


def deterministic_variant(idx: str, rec: dict | None = None) -> str:
    """ref_edit (com referencia) ou tts_instruction (sem), por hash do idx.
    ref_edit_auto usa locutor (ref_map/pool); Tata e fallback ficam self-ref."""
    h = hashlib.sha1(idx.encode()).digest()[0]
    if (h / 255.0) >= REF_EDIT_FRAC:
        return "tts_instruction"
    if rec is not None and rec.get("speaker") and rec["speaker"] != "tata":
        return "ref_edit_auto"
    return "ref_edit_tata"


def instruction_for_rec(rec: dict) -> str:
    """Instrucao deterministica por amostra (mistura hash do idx + frames)."""
    seed = (int(rec["frames"]) * 7 + sum(map(ord, rec["idx"]))) % 10_000
    return CB.INSTRUCTION_POOL[seed % len(CB.INSTRUCTION_POOL)]


_SPK_POOL: dict[str, list[str]] | None = None


def _speaker_pool() -> dict[str, list[str]]:
    """speaker -> idxs DO SPLIT DE TREINO (evita vazamento de val/test como ref)."""
    global _SPK_POOL
    if _SPK_POOL is not None:
        return _SPK_POOL
    train = set(load_split_file("train"))
    pool: dict[str, list[str]] = {}
    for r in load_meta():
        if r["idx"] in train:
            pool.setdefault(r.get("speaker", "tata"), []).append(r["idx"])
    _SPK_POOL = pool
    return _SPK_POOL


def load_split_file(name: str) -> list[str]:
    p = CB.TRAINING / f"splits_{name}.txt"
    if not p.exists():
        return []
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _rec_by_idx() -> dict[str, dict]:
    """Mapa idx->rec (evita busca linear O(N) por item ref_edit)."""
    global _REC_BY_IDX
    if _REC_BY_IDX is None:
        _REC_BY_IDX = {r["idx"]: r for r in load_meta()}
    return _REC_BY_IDX


_REC_BY_IDX: dict[str, dict] | None = None


def pick_ref_rec(rec: dict) -> dict:
    """Referencia do MESMO locutor. Prioridade: ref_map (idx->ref_idx com guarda
    de similaridade); senao pool por speaker + hash; senao self-ref."""
    ridx = rec.get("ref_idx")
    if ridx:
        rr = _rec_by_idx().get(ridx)
        if rr is not None:
            return rr
    spk = rec.get("speaker", "tata")
    pool = [i for i in _speaker_pool().get(spk, []) if i != rec["idx"]]
    if not pool:
        return rec
    j = int(hashlib.sha1(("ref:" + rec["idx"]).encode()).hexdigest(), 16) % len(pool)
    return _rec_by_idx().get(pool[j], rec)


def build_item(tokenizer, rec: dict, variant: str) -> dict[str, object]:
    """Item pronto p/ DataLoader (tensores 1-D). Usa cache de codes.

    ref_edit_auto: ref = OUTRO clipe do MESMO locutor (fallback self-ref se o
    locutor so tem 1 clipe no treino). Template/policies = ref_edit_tata.
    """
    wav_key = str(CB.WAVS24_DIR / f"{rec['idx']}.wav")
    n_frames = CB.register_codes(wav_key, CB.TOKENS_DIR / f"{rec['idx']}.npz")
    instruction = instruction_for_rec(rec)
    if variant == "ref_edit_auto":
        ref_rec = pick_ref_rec(rec)
        ref_key = str(CB.WAVS24_DIR / f"{ref_rec['idx']}.wav")
        n_ref = CB.register_codes(ref_key, CB.TOKENS_DIR / f"{ref_rec['idx']}.npz")
        req = CB.ExampleRequest(
            variant="ref_edit_tata",
            text=rec["text"],
            instruction=instruction,
            ref_text=ref_rec["text"],
            ref_audio_path=ref_key,
            target_audio_path=wav_key,
        )
    else:
        n_ref = 0
        req = CB.ExampleRequest(
            variant=variant,
            text=rec["text"],
            instruction=instruction,
            ref_text=rec["text"],
            ref_audio_path=wav_key,
        )
    ex = CB.build_example(tokenizer, req)
    ex["labels"] = CB.make_labels(ex, frame_policies=CB.POLICIES_BY_VARIANT[variant])
    # sanity estrutural
    assert len(ex["input_ids"][0]) == len(ex["labels"][0])
    n_audio_placeholders = int((ex["input_ids"][0] == CB.AUDIO_TOKEN_ID).sum())
    if variant == "ref_edit_auto":
        expected_frames = n_frames + n_ref
    else:
        expected_frames = n_frames * (2 if variant == "ref_edit_tata" else 1)
    assert n_audio_placeholders == expected_frames, (n_audio_placeholders, expected_frames)
    ex["_n_frames"] = n_frames
    ex["_variant"] = variant
    ex["_instruction"] = instruction
    return ex


def finalize(num_gold: int, parity_device: str) -> None:
    from breeze_infer.templates import _prepare_one

    recs = load_meta()
    print(f"[finalize] registros={len(recs)}")
    assert len(recs) >= 100, "rode o 'process' completo antes do finalize"

    # split ESTRATIFICADO POR CORPUS (train/val/test em cada fonte de dados)
    cuts: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    by_corpus: dict[str, list[str]] = {}
    for r in recs:
        by_corpus.setdefault(r.get("corpus", "tata"), []).append(r["idx"])
    for corpus, lst in by_corpus.items():
        idxs = sorted(lst)
        rng = __import__("random").Random(42)
        rng.shuffle(idxs)
        n = len(idxs)
        if n < 40:
            cuts["train"].extend(idxs)
            print(f"[finalize] corpus '{corpus}': {n} itens -> train (pequeno)")
            continue
        cuts["train"].extend(idxs[: int(n * 0.9)])
        cuts["val"].extend(idxs[int(n * 0.9): int(n * 0.95)])
        cuts["test"].extend(idxs[int(n * 0.95):])
        print(f"[finalize] corpus '{corpus}': train={int(n*0.9)} val/test={n - int(n*0.9)}")
    for name, lst in cuts.items():
        (CB.TRAINING / f"splits_{name}.txt").write_text("\n".join(lst), encoding="utf-8")
    print({k: len(v) for k, v in cuts.items()})

    tokenizer = CB.load_text_tokenizer()

    # ---- paridade com codigo oficial (sem cache): referencia codifica o wav24 real
    # (gold na Tata = corpus de leitura com paridade ja validada na Fase B)
    rec_by_idx = {r["idx"]: r for r in recs}
    g_idx = cuts["val"][: max(1, num_gold // 2)] + cuts["test"][: max(1, num_gold - num_gold // 2)]
    gold_recs = [rec_by_idx[i] for i in g_idx if rec_by_idx[i].get("corpus", "tata") == "tata"]
    CB.GOLD_DIR.mkdir(parents=True, exist_ok=True)

    parity_fail = 0
    for gi, rec in enumerate(gold_recs):
        wav_key = str(CB.WAVS24_DIR / f"{rec['idx']}.wav")
        atok_official = CB.load_audio_tokenizer(parity_device)
        for variant in ("tts_instruction", "ref_edit_tata"):
            ex = build_item(tokenizer, rec, variant)
            instruction = ex["_instruction"]

            # ---- paridade: (1) prefixo oficial puro; (2) bloco alvo isolado

            base_req = CB.ExampleRequest(
                variant=variant, text=rec["text"], instruction=instruction,
                ref_text=rec["text"], ref_audio_path=wav_key)
            prefix_official = _prepare_one(
                tokenizer, atok_official, CB.ConfigStub(16),
                CB.build_segments(base_req, include_target=False))
            target_only = _prepare_one(
                tokenizer, atok_official, CB.ConfigStub(16),
                [{"type": "audio", "append_eos": True, "drop_last_frame": False,
                  "audio_path": wav_key}])

            our_ids = ex["input_ids"][0]
            pre_len = prefix_official["input_ids"].shape[1]
            n_tgt_frames = CB.count_frames_per_block(ex)[-1]
            same_prefix = bool((ex["input_ids"][:, :pre_len] == prefix_official["input_ids"]).all())
            tgt_ids_official = target_only["input_ids"][0]
            same_suffix = bool((our_ids[pre_len:] == tgt_ids_official).all())
            our_tail = ex["input_values"][0, -n_tgt_frames:, :]
            official_tail = target_only["audio_tokens"][0]
            same_vals = bool((our_tail.long() == official_tail.long()).all())
            ok = same_prefix and same_suffix and same_vals
            parity_fail += 0 if ok else 1

            lab = ex["labels"][0]
            vals, cnts = np.unique(lab.numpy(), return_counts=True)
            dist = {str(int(v)): int(c) for v, c in zip(vals, cnts)}
            block_counts = CB.count_frames_per_block(ex)
            fname = CB.GOLD_DIR / f"gold{gi}_{variant}.txt"
            with fname.open("w", encoding="utf-8") as fh:
                fh.write(f"# GOLD {gi} variant={variant}\n")
                fh.write(f"idx={rec['idx']}  dur={rec['dur_proc_s']}s  frames={rec['frames']}\n")
                fh.write(f"instruction={instruction}\n\n")
                fh.write(f"text={rec['text']}\n\n")
                fh.write(f"seq_len={len(our_ids)} blocks_frames={block_counts} "
                         f"(prefixo_oficial={block_counts[:-1] if len(block_counts)>1 else []}, "
                         f"alvo=ultimo)\n")
                fh.write(f"label_dist={dist}\n")
                dec = tokenizer.decode(ex["input_ids"][0].tolist(), skip_special_tokens=False)
                fh.write(f"input_ids_preview={dec[:400]}\n")
                fh.write(f"parity(prefixo,sufixo,values) vs codigo oficial cru = "
                         f"({same_prefix},{same_suffix},{same_vals})\n")
            torch_save(ex, CB.GOLD_DIR / f"gold{gi}_{variant}.pt")
    print(f"[finalize] paridade FALHOU em {parity_fail}/{len(gold_recs)*2}")

    durations = [r["dur_proc_s"] for r in recs]
    frames_all = [r["frames"] for r in recs]
    by_corpus_stats = {}
    for r in recs:
        c = r.get("corpus", "tata")
        d = by_corpus_stats.setdefault(c, {"n": 0, "s": 0.0})
        d["n"] += 1
        d["s"] += r["dur_proc_s"]
    summary = {
        "total_kept": len(recs),
        "hours_kept": round(sum(durations) / 3600, 2),
        "avg_dur_s": round(sum(durations) / len(recs), 2),
        "max_dur_s": max(durations),
        "frames_avg": round(sum(frames_all) / len(frames_all)),
        "frames_max": max(frames_all),
        "by_corpus": {c: {"n": d["n"], "hours": round(d["s"] / 3600, 2)}
                      for c, d in by_corpus_stats.items()},
        "gold_parities_failed": parity_fail,
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[finalize]", json.dumps(summary, ensure_ascii=False))

    manifest_lines = ["idx,corpus,speaker,wav_rel,text,dur_src_s,dur_proc_s,frames,variant"]
    for r in recs:
        safe = r["text"].replace('"', "'")
        manifest_lines.append(
            f"{r['idx']},{r.get('corpus','tata')},{r.get('speaker','tata')},{r['wav_rel']},"
            f"\"{safe}\",{r['dur_src_s']},{r['dur_proc_s']},{r['frames']},"
            f"{deterministic_variant(r['idx'], r)}"
        )
    MANIFEST_CSV.write_text("\n".join(manifest_lines), encoding="utf-8")


def torch_save(ex: dict, path: Path) -> None:
    slim = {
        "input_ids": ex["input_ids"], "attention_mask": ex.get("attention_mask"),
        "text_ids_mask": ex["text_ids_mask"], "text_ids_len": ex["text_ids_len"],
        "input_values": ex["input_values"].int(), "labels": ex["labels"],
        "_variant": ex["_variant"], "_n_frames": ex["_n_frames"],
    }
    torch.save(slim, path)


# ------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["process", "finalize"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gold", type=int, default=3)
    ap.add_argument("--parity-device", default="cuda")
    args = ap.parse_args()

    sys.path.insert(0, str(CB.REPO))
    if args.cmd == "process":
        process(args.limit, args.device)
    else:
        finalize(args.gold, args.parity_device)


if __name__ == "__main__":
    main()
