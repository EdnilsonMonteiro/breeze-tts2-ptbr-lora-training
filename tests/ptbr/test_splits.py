from splits import audit_leaks, group_key, make_splits


def _recs():
    recs = []
    # tagarela: 40 programas x 3 "locutores" (show:0, show:1, show:2), 10 clipes cada
    for s in range(40):
        for k in range(3):
            for c in range(10):
                recs.append({"idx": f"t{s}_{k}_{c}", "corpus": "tagarela",
                             "speaker": f"show_{s}:{k}", "dur_proc_s": 5.0})
    # cetuc: 20 locutores x 30
    for s in range(20):
        for c in range(30):
            recs.append({"idx": f"c{s}_{c}", "corpus": "cetuc", "speaker": f"cetuc:S{s}",
                         "dur_proc_s": 5.0})
    # tata: 1 locutor
    for c in range(100):
        recs.append({"idx": f"tt{c}", "corpus": "tata", "speaker": "tata", "dur_proc_s": 5.0})
    return recs


def test_group_key_collapses_show_speakers():
    assert group_key({"idx": "x", "corpus": "tagarela", "speaker": "show_1:3"}) == "tagarela/show_1"
    assert group_key({"idx": "x", "corpus": "cetuc", "speaker": "cetuc:A"}) == "cetuc/cetuc:A"


def test_no_group_or_speaker_leak_and_deterministic():
    recs = _recs()
    sp = make_splits(recs, seed=42)
    a = audit_leaks(recs, sp)
    assert a == {"group_leaks": [], "speaker_leaks": []}
    assert sp == make_splits(recs, seed=42)
    assert sorted(sum(sp.values(), [])) == sorted(r["idx"] for r in recs)   # particao completa


def test_fractions_and_nonempty_val_test():
    recs = _recs()
    sp = make_splits(recs)
    n = len(recs)
    assert 0.85 < len(sp["train"]) / n < 0.95
    assert len(sp["val"]) > 0 and len(sp["test"]) > 0
    tata = {r["idx"] for r in recs if r["corpus"] == "tata"}
    assert tata <= set(sp["train"])           # corpus de 1 locutor fica todo no treino


def test_audit_detects_old_style_leak():
    recs = _recs()
    bad = {"train": [r["idx"] for r in recs if r["speaker"] == "show_0:0"],
           "val": [r["idx"] for r in recs if r["speaker"] == "show_0:1"], "test": []}
    assert audit_leaks(recs, bad)["group_leaks"] == ["tagarela/show_0"]
