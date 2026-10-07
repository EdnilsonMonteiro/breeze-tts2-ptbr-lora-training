import importlib.util
import sys
from pathlib import Path

import numpy as np

_P = Path(__file__).resolve().parents[2] / "ptbr_lora" / "data" / "ingest_extra_corpus.py"
_spec = importlib.util.spec_from_file_location("ingest_extra_corpus", _P)
IN = importlib.util.module_from_spec(_spec)
sys.modules["ingest_extra_corpus"] = IN
_spec.loader.exec_module(IN)


def test_parse_cv_filters_votes_and_names(tmp_path):
    (tmp_path / "clips").mkdir()
    hdr = "client_id\tpath\tsentence\tup_votes\tdown_votes\n"
    rows = ["abcd1234abcd1234ZZ\tcommon_voice_pt_1.mp3\tOlá, tudo bem?\t2\t0",
            "abcd1234abcd1234ZZ\tcommon_voice_pt_2.mp3\tFrase com voto ruim.\t2\t1",
            "abcd1234abcd1234ZZ\tcommon_voice_pt_3.mp3\tPoucos votos.\t1\t0",
            "ffff0000ffff0000YY\tcommon_voice_pt_4.mp3\t\t3\t0"]
    (tmp_path / "validated.tsv").write_text(hdr + "\n".join(rows) + "\n", encoding="utf-8")
    out = IN.parse_cv(tmp_path)
    assert [r["idx"] for r in out] == ["cv_pt_common_voice_pt_1"]
    assert out[0]["speaker"] == "cv_pt:abcd1234abcd1234" and out[0]["text"] == "Olá, tudo bem?"
    assert out[0]["src"].endswith("clips" + __import__("os").sep + "common_voice_pt_1.mp3")


def test_parse_mls(tmp_path):
    d = tmp_path / "train"
    (d / "audio" / "1234" / "5678").mkdir(parents=True)
    (d / "audio" / "1234" / "5678" / "1234_5678_000001.flac").write_bytes(b"x")
    (d / "transcripts.txt").write_text("1234_5678_000001\tum texto qualquer\nlinha_ruim\n", encoding="utf-8")
    out = IN.parse_mls(tmp_path)
    assert len(out) == 1 and out[0]["speaker"] == "mls_pt:1234" and out[0]["src"].endswith(".flac")


def test_limit_per_speaker_min_max_deterministic():
    rows = ([{"idx": f"a{i}", "speaker": "A"} for i in range(50)]
            + [{"idx": f"b{i}", "speaker": "B"} for i in range(5)]
            + [{"idx": f"c{i}", "speaker": "C"} for i in range(20)])
    out = IN.limit_per_speaker(rows, min_per=14, max_per=30)
    spk = [r["speaker"] for r in out]
    assert spk.count("A") == 30 and spk.count("C") == 20 and "B" not in spk
    assert out == IN.limit_per_speaker(rows, min_per=14, max_per=30)


def test_snr_estimate_orders_clean_vs_noisy():
    sr = 16000
    rng = np.random.default_rng(0)
    t = np.arange(sr * 4) / sr
    speech = np.sin(2 * np.pi * 150 * t) * (np.sin(2 * np.pi * 2.5 * t) > 0)       # rajadas + pausas
    clean = (0.3 * speech + 1e-4 * rng.standard_normal(len(t))).astype("float32")
    noisy = (0.3 * speech + 0.05 * rng.standard_normal(len(t))).astype("float32")
    assert IN.estimate_snr_db(clean, sr) > 30
    assert IN.estimate_snr_db(noisy, sr) < IN.estimate_snr_db(clean, sr) - 15
    assert IN.quality_ok(clean, sr, 18.0) and not IN.quality_ok(noisy, sr, 18.0)
    assert IN.quality_ok(noisy, sr, 0.0)                                              # filtro desligado


def _hf_tree(tmp_path, spk_specs):
    """spk_specs: [(client_id, gender, n_clips, split)] -> root com transcript/pt/<split>.tsv."""
    (tmp_path / "transcript" / "pt").mkdir(parents=True)
    by_split = {}
    k = 0
    for cid, g, n, sp in spk_specs:
        for _ in range(n):
            by_split.setdefault(sp, []).append(
                f"{cid}\tcommon_voice_pt_{k}.mp3\tFrase numero {k} do corpus.\t2\t0\t\t{g}\t\tpt\t\t")
            k += 1
    for sp, lines in by_split.items():
        hdr = "client_id\tpath\tsentence\tup_votes\tdown_votes\tage\tgender\taccent\tlocale\tsegment\tvariant\n"
        (tmp_path / "transcript" / "pt" / f"{sp}.tsv").write_text(hdr + "\n".join(lines) + "\n", encoding="utf-8")


def test_normalize_gender():
    assert IN.normalize_gender("female_feminine") == "F" and IN.normalize_gender("male_masculine") == "M"
    assert IN.normalize_gender("female") == "F" and IN.normalize_gender("male") == "M"
    assert IN.normalize_gender("") is None and IN.normalize_gender(None) is None
    assert IN.normalize_gender("non-binary") is None and IN.normalize_gender("do_not_wish_to_say") is None


def test_parse_cv_hf_and_select_balance_caps(tmp_path):
    specs = ([(f"fem{i:02d}xxxxxxxxxxxxxx", "female_feminine", 20, "train") for i in range(4)]
             + [(f"mal{i:02d}xxxxxxxxxxxxxx", "male_masculine", 20, "train") for i in range(9)]
             + [("poucoxxxxxxxxxxxxxx", "male_masculine", 5, "dev"),               # < min_per
                ("semgenxxxxxxxxxxxxxx", "", 30, "test")])                         # genero desconhecido
    _hf_tree(tmp_path, specs)
    rows = IN.parse_cv_hf(tmp_path)
    assert len(rows) == 4 * 20 + 9 * 20 + 5 + 30
    assert all(r["src"].endswith(".mp3") and r["fname"].endswith(".mp3") for r in rows)
    sel, st = IN.select_cv_candidates(rows, min_per=12, max_per=15, max_voices_per_gender=6)
    assert st["eligible_F"] == 4 and st["eligible_M"] == 9
    assert st["selected_F"] == 4 and st["selected_M"] == 6                          # teto por genero
    assert st["unknown_gender_speakers"] == 1
    per = {}
    for r in sel:
        per[r["speaker"]] = per.get(r["speaker"], 0) + 1
    assert max(per.values()) == 15 and len(per) == 10
    assert all(r["gender"] in ("F", "M") for r in sel)
    sel2, _ = IN.select_cv_candidates(rows, 12, 15, 6)
    assert [r["idx"] for r in sel] == [r["idx"] for r in sel2]                      # deterministico


def test_extract_wanted_from_tar(tmp_path):
    import io
    import tarfile

    tar_p = tmp_path / "pt_train_0.tar"
    with tarfile.open(tar_p, "w") as tf:
        for name in ("pt_train_0/a.mp3", "pt_train_0/b.mp3", "c.mp3"):
            data = name.encode()
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    n = IN.extract_wanted(tar_p, {"a.mp3", "c.mp3", "zzz.mp3"}, tmp_path / "clips")
    assert n == 2
    assert sorted(p.name for p in (tmp_path / "clips").iterdir()) == ["a.mp3", "c.mp3"]
    assert IN.extract_wanted(tar_p, {"a.mp3"}, tmp_path / "clips") == 0             # idempotente
