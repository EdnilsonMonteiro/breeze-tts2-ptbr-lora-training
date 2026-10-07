import random

import voices as V
from splits import audit_leaks, group_key


def _recs(seed=0):
    rnd = random.Random(seed)
    recs = []

    def add(corpus, spk, n, prefix):
        for c in range(n):
            recs.append({"idx": f"{prefix}_{c}", "corpus": corpus, "speaker": spk,
                         "dur_proc_s": round(rnd.uniform(3.0, 9.5), 2), "text": "uma frase de teste valida"})

    for s in range(60):                                   # tagarela: 60 shows x 3 clusters
        for k in range(3):
            add("tagarela", f"show{s}:{k}", 30 + 5 * k, f"tg{s}_{k}")
    for k in range(5):                                    # clipes de locutor nao atribuido
        add("tagarela", f"show{k}:99", 50, f"un{k}")
    for s in range(10):                                   # podcast: 10 shows x 4
        for k in range(4):
            add("podcast", f"pod{s}:{k}", 50, f"pd{s}_{k}")
    for s in range(30):
        add("cetuc", f"cetuc:S{s}", 200, f"ce{s}")
    for s in range(6):                                    # cml: 6 enormes + 14 pequenos
        add("cml_pt", f"cml:{s}", 2000, f"cmb{s}")
    for s in range(14):
        add("cml_pt", f"cml:s{s}", 25 if s < 10 else 5, f"cms{s}")
    add("tata", "tata", 300, "tt")
    return recs


def test_select_filters_and_caps():
    cfg = V.VoiceCfg()
    recs = _recs()
    recs.append({"idx": "curto", "corpus": "cetuc", "speaker": "cetuc:S0", "dur_proc_s": 1.0,
                 "text": "frase ok valida"})
    sel = V.select_voices(recs, cfg)
    by = {}
    for r in sel:
        by.setdefault(V.voice_key(r), []).append(r)
    assert all(len(v) >= cfg.min_clips for v in by.values())
    assert not any(str(r["speaker"]).endswith(":99") for r in sel)
    assert all(cfg.min_dur_s <= r["dur_proc_s"] <= cfg.max_dur_s for r in sel)
    assert max(len(v) for k, v in by.items() if k[0] == "cml_pt") == 120
    assert max(len(v) for k, v in by.items() if k[0] == "tagarela") == 40
    assert not [k for k in by if k[0] == "cml_pt" and k[1] in {"cml:s10", "cml:s11", "cml:s12", "cml:s13"}]  # 5 clipes
    assert V.select_voices(recs, cfg) == sel                                           # deterministico


def test_splits_group_disjoint_and_caps():
    cfg = V.VoiceCfg()
    sel = V.select_voices(_recs(), cfg)
    sp = V.make_voice_splits(sel, cfg)
    leaks = audit_leaks(sel, sp)
    assert leaks == {"group_leaks": [], "speaker_leaks": []}
    by_idx = {r["idx"]: r for r in sel}
    summ = V.split_summary(sel, sp)
    assert summ["val"]["cetuc"]["groups"] == 3 and summ["test"]["cetuc"]["groups"] == 3
    assert summ["val"]["tagarela"]["groups"] == 16 and summ["test"]["tagarela"]["groups"] == 16
    assert "tata" not in summ["val"]                                                  # tata so treina
    per = {}
    for s in ("val", "test"):
        for i in sp[s]:
            per[(s, V.voice_key(by_idx[i]))] = per.get((s, V.voice_key(by_idx[i])), 0) + 1
    assert max(per.values()) <= cfg.heldout_cap
    assert V.make_voice_splits(sel, cfg) == sp
    assert set(sp["train"]).isdisjoint(sp["val"]) and set(sp["train"]).isdisjoint(sp["test"])


def test_mass_weights_caps_and_normalisation():
    cfg = V.VoiceCfg()
    sel = V.select_voices(_recs(), cfg)
    sp = V.make_voice_splits(sel, cfg)
    by_idx = {r["idx"]: r for r in sel}
    tr = [by_idx[i] for i in sp["train"]]
    w, info = V.speaker_mass_weights(tr, cap_spk=0.01, cap_grp=0.02)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert not info["relaxed"]
    assert info["max_voice_mass"] <= 0.01 + 1e-6
    assert info["max_group_mass"] <= 0.02 + 1e-6
    assert all(x > 0 for x in w.values())
    # a participacao por corpus segue o `share` quando os tetos permitem
    cm = info["corpus_mass"]
    assert abs(cm["tagarela"] - 0.50 / 0.97) < 0.02 or cm["tagarela"] > 0.3
    # voz com mais clipes NAO domina (CML enorme: 120 clipes, mesmo teto de qualquer outra)
    assert info["eff_voices"] > 0.6 * info["n_voices"] * 0.5


def test_mass_weights_relaxed_when_infeasible():
    items = [{"idx": f"a{i}_{c}", "corpus": "cetuc", "speaker": f"cetuc:S{i}"}
             for i in range(10) for c in range(20)]
    w, info = V.speaker_mass_weights(items, cap_spk=0.01, cap_grp=0.02)
    assert info["relaxed"] and abs(sum(w.values()) - 1.0) < 1e-9
    assert info["max_voice_mass"] < 0.1 + 1e-3                         # ~ 1/10 (uniforme)


def test_beta_zero_is_uniform_per_voice():
    items = ([{"idx": f"a{c}", "corpus": "cetuc", "speaker": "cetuc:A"} for c in range(100)]
             + [{"idx": f"b{c}", "corpus": "cetuc", "speaker": "cetuc:B"} for c in range(10)])
    w, info = V.speaker_mass_weights(items, cap_spk=1.0, cap_grp=1.0, beta=0.0)
    ma = sum(w[f"a{c}"] for c in range(100))
    mb = sum(w[f"b{c}"] for c in range(10))
    assert abs(ma - mb) < 1e-9


def test_pick_eval_voices_from_split_and_prompt_ok():
    cfg = V.VoiceCfg()
    sel = V.select_voices(_recs(), cfg)
    sp = V.make_voice_splits(sel, cfg)
    by_idx = {r["idx"]: r for r in sel}
    ev = V.pick_eval_voices(by_idx, sp["val"], n=4)
    assert len(ev) == 4
    assert [e["corpus"] for e in ev] == ["tagarela", "podcast", "cml_pt", "cetuc"]
    val = set(sp["val"])
    for e in ev:
        assert e["prompt_idx"] in val and set(e["heldout_idx"]) <= val
        assert e["prompt_idx"] not in e["heldout_idx"] and e["heldout_idx"]
        d = by_idx[e["prompt_idx"]]["dur_proc_s"]
        assert 4.0 <= d <= 9.0
        spk = {V.voice_key(by_idx[i]) for i in [e["prompt_idx"], *e["heldout_idx"]]}
        assert len(spk) == 1
    assert V.pick_eval_voices(by_idx, sp["val"], n=4) == ev


def test_normalized_secs():
    assert abs(V.normalized_secs(0.6, 0.8, 0.2) - (0.4 / 0.6)) < 1e-9
    assert V.normalized_secs(0.6, 0.2, 0.2) != V.normalized_secs(0.6, 0.2, 0.2)    # nan


def test_max_repeat_caps_small_voices():
    cfg = V.VoiceCfg()
    sel = V.select_voices(_recs(), cfg)
    sp = V.make_voice_splits(sel, cfg)
    by_idx = {r["idx"]: r for r in sel}
    tr = [by_idx[i] for i in sp["train"]]
    S = int(5 * len(tr))                                        # soma dos tetos = 1,2 (viavel)
    w, info = V.speaker_mass_weights(tr, cap_spk=0.05, cap_grp=0.2, total_samples=S, max_repeat=6.0)
    assert abs(sum(w.values()) - 1.0) < 1e-9 and not info["relaxed"]
    assert info["max_repeat_obs"] <= 6.0 + 1e-3                # nenhum clipe visto >6x em media
    w0, info0 = V.speaker_mass_weights(tr, cap_spk=0.05, cap_grp=0.2)   # sem teto de repeticao
    n = {}
    for r in tr:
        n[V.voice_key(r)] = n.get(V.voice_key(r), 0) + 1
    rep0 = max(sum(w0[r["idx"]] for r in tr if V.voice_key(r) == k) * S / n[k] for k in n)
    assert rep0 > 6.0
