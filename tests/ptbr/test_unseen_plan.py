import json
import sys
import types

import numpy as np
import soundfile as sf

import unseen_eval as UE


def _recs():
    recs = {}
    for c, spk in (("tagarela", "tagarela:show1:1"), ("cml_pt", "cml_pt:A")):
        for i in range(8):
            idx = f"{c}_{i}"
            recs[idx] = {"idx": idx, "corpus": c, "speaker": spk, "text": f"texto {idx}",
                         "dur_proc_s": 5.0, "group": f"{c}/{spk}"}
    return recs


def _wav(path, f=220.0, sr=16000, dur=1.0):
    t = np.arange(int(sr * dur)) / sr
    sf.write(str(path), (0.1 * np.sin(2 * np.pi * f * t)).astype("float32"), sr)


def _user(tmp_path, names=("VozA", "VozB", "VozC")):
    ex = []
    for k, n in enumerate(names):
        p = tmp_path / f"{n}.wav"
        _wav(p, 200 + 80 * k)
        ex.append({"name": n, "audio": str(p), "text": f"ref {n}"})
    return ex


def test_plan_has_val_and_three_user_voices(tmp_path):
    recs = _recs()
    plan = UE.build_plan(recs, list(recs), tmp_path, n_voices=2, extra=_user(tmp_path))
    groups = [(v["name"], v["group"]) for v in plan["voices"]]
    assert [g for _, g in groups] == ["val", "val", "user", "user", "user"]
    assert [n for n, g in groups if g == "user"] == ["VozA", "VozB", "VozC"]
    u = [v for v in plan["voices"] if v["group"] == "user"]
    assert all(v["self_heldout"] and v["heldout"] == [v["prompt"]["audio"]] for v in u)


def test_single_dict_extra_still_works_and_duplicate_names_rejected(tmp_path):
    recs = _recs()
    ex = _user(tmp_path, ("VozA",))[0]
    plan = UE.build_plan(recs, list(recs), tmp_path, n_voices=1, extra=ex)
    assert plan["voices"][-1]["name"] == "VozA"
    import pytest
    with pytest.raises(ValueError):
        UE.build_plan(recs, list(recs), tmp_path, n_voices=0, extra=_user(tmp_path, ("A", "A")))


def test_jobs_user_gets_all_phrases_val_only_selected(tmp_path):
    recs = _recs()
    plan = UE.build_plan(recs, list(recs), tmp_path, n_voices=2, extra=_user(tmp_path))
    jobs = UE.jobs_for(plan, 2, val_phrase_ids=[1])
    names = [j[0] for j in jobs]
    assert sum(n.startswith(("VozA", "VozB", "VozC")) for n in names) == 6
    val = [n for n in names if not n.startswith(("VozA", "VozB", "VozC"))]
    assert len(val) == 2 and all(n.endswith("__p1") for n in val)
    assert len(UE.jobs_for(plan, 2)) == 10                      # sem filtro: tudo (compatibilidade)


def test_score_group_means_with_fake_metrics(tmp_path, monkeypatch):
    recs = _recs()
    plan = UE.build_plan(recs, list(recs), tmp_path, n_voices=2, extra=_user(tmp_path))
    for v in plan["voices"]:                                    # wavs de referencia/heldout dos vals
        for p in [v["prompt"]["audio"], *v["heldout"]]:
            _wav(tmp_path / p.split("/")[-1]) if not __import__("pathlib").Path(p).exists() else None
    out = tmp_path / "out"
    out.mkdir()
    for j in UE.jobs_for(plan, 2, val_phrase_ids=[1]):
        _wav(out / f"{j[0]}.wav", 300)

    fake = types.ModuleType("metrics")
    fake.embed_array = lambda y, dev="cpu": np.array([1.0, float(np.mean(np.abs(y))), 0.0], dtype=np.float32)
    fake.cos = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    fake.transcribe = lambda f, size, dev, ct: "texto"
    fake.wer_cer = lambda ref, hyp: (0.25, 0.1)
    monkeypatch.setitem(sys.modules, "metrics", fake)
    res = UE.score(out, plan, 2)
    assert res["n"] == 8                                       # 3 voz usuario x 2 + 2 val x 1
    assert res["wer_user"] == 0.25 and res["wer_val"] == 0.25
    assert 0.0 <= res["secs_user"] <= 1.0 and 0.0 <= res["secs_val"] <= 1.0
    assert {i["group"] for i in res["items"]} == {"val", "user"}
    csvp = tmp_path / "e.csv"
    UE.append_csv(csvp, 500, res)
    header = csvp.read_text().splitlines()[0].split(",")
    assert header[-4:] == ["secs_val", "wer_val", "secs_user", "wer_user"]
    assert len(csvp.read_text().splitlines()[1].split(",")) == len(header)
