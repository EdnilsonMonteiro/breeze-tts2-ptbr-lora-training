"""meta_io.py — leitura de corpora e metadados SEM torch (usado por split/resplit/testes).

Extraido de `prepare_dataset.py` / `common_breeze.py` (que importam torch no topo) para que
o split e as auditorias rodem em qualquer maquina.
"""
from __future__ import annotations

import json
from pathlib import Path

import paths

_LEGACY_CORPORA = [
    {"name": "tata", "root": "TTS-Portuguese-Corpus", "csv": "texts.csv"},
    {"name": "podcast", "root": "podcast", "csv": "texts.csv"},
]


def load_corpora() -> list[dict]:
    if paths.CORPORA_JSON.exists():
        data = json.loads(paths.CORPORA_JSON.read_text(encoding="utf-8"))
        return [c for c in data if c.get("enabled", True)]
    return list(_LEGACY_CORPORA)


def corpus_dir(corp: dict) -> Path:
    return paths.DATASETS_ROOT / corp["root"]


def corpus_csv(corp: dict) -> Path:
    return corpus_dir(corp) / corp.get("csv", "texts.csv")


def _jsonl(path: Path):
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            yield json.loads(ln)


def chunk_speaker_map() -> dict[str, str]:
    """idx do podcast legado -> 'tag:SPEAKER_XX'."""
    ci = paths.SCRAPING_WORK / "chunks_index.jsonl"
    if not ci.exists():
        return {}
    return {r["chunk"].removesuffix(".wav"): f"{r['tag']}:{r['speaker']}" for r in _jsonl(ci)}


def apply_corpus_overrides(by_idx: dict[str, dict]) -> None:
    """speakers.jsonl (locutor) e ref_map.jsonl (ref_idx legado) de cada corpus."""
    for corp in load_corpora():
        root = corpus_dir(corp)
        spk_file = root / "speakers.jsonl"
        if spk_file.exists():
            for r in _jsonl(spk_file):
                t = by_idx.get(r["idx"])
                if t is not None:
                    t["speaker"] = r["speaker"]
                    t.setdefault("corpus", corp["name"])
                    if r.get("group"):                  # unidade de split explicita (ex.: podcast_identity)
                        t["group"] = r["group"]
        ref_file = root / "ref_map.jsonl"
        if ref_file.exists():
            for r in _jsonl(ref_file):
                t = by_idx.get(r["idx"])
                if t is not None and r.get("ref_idx"):
                    t["ref_idx"] = r["ref_idx"]


def load_meta_records(meta_jsonl: Path | None = None) -> list[dict]:
    """dataset_meta.jsonl + overrides; ultima ocorrencia de cada idx vence."""
    meta_jsonl = meta_jsonl or (paths.TRAINING / "dataset_meta.jsonl")
    by_idx: dict[str, dict] = {}
    for r in _jsonl(meta_jsonl):
        by_idx[r["idx"]] = r
    recs = list(by_idx.values())
    spk = chunk_speaker_map()
    for r in recs:
        if "speaker" not in r:
            r["corpus"] = "podcast" if r["idx"] in spk else "tata"
            r["speaker"] = spk.get(r["idx"], "tata")
    apply_corpus_overrides(by_idx)
    return recs
