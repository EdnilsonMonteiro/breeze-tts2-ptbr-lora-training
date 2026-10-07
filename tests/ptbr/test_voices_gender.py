import voices as V
from splits import audit_leaks


def _cv(nf=10, nm=25, nu=6, clips=20):
    recs = []
    for g, n, tag in (("female_feminine", nf, "f"), ("male_masculine", nm, "m"), ("", nu, "u")):
        for s in range(n):
            for c in range(clips):
                recs.append({"idx": f"cv_{tag}{s}_{c}", "corpus": "cv_pt", "speaker": f"cv_pt:{tag}{s}",
                             "gender": g, "dur_proc_s": 4.0 + (c % 5), "text": "uma frase de teste valida"})
    # outro corpus sem genero: deve passar intacto
    for s in range(15):
        for c in range(20):
            recs.append({"idx": f"ce{s}_{c}", "corpus": "cetuc", "speaker": f"cetuc:S{s}",
                         "dur_proc_s": 5.0, "text": "uma frase de teste valida"})
    return recs


def test_gender_of_variants():
    assert V.gender_of({"gender": "female_feminine"}) == "F" and V.gender_of({"gender": "Female"}) == "F"
    assert V.gender_of({"gender": "male_masculine"}) == "M" and V.gender_of({"gender": "male"}) == "M"
    assert V.gender_of({"gender": ""}) is None and V.gender_of({}) is None
    assert V.gender_of({"gender": "do_not_wish_to_say"}) is None


def test_select_balances_voices_mf_and_drops_unknown():
    cfg = V.VoiceCfg()
    sel = V.select_voices(_cv(), cfg)
    cv = [r for r in sel if r["corpus"] == "cv_pt"]
    voices = {V.voice_key(r) for r in cv}
    f = {v for v in voices if v[1].startswith("cv_pt:f")}
    m = {v for v in voices if v[1].startswith("cv_pt:m")}
    assert len(f) == len(m) == 10                        # min(10, 25)
    assert not any(v[1].startswith("cv_pt:u") for v in voices)  # genero desconhecido sai
    assert len([r for r in sel if r["corpus"] == "cetuc"]) == 15 * 20   # outros corpora intactos
    assert V.select_voices(_cv(), cfg) == sel            # deterministico


def test_splits_balanced_by_gender_and_no_leak():
    cfg = V.VoiceCfg()
    sel = V.select_voices(_cv(nf=20, nm=40), cfg)
    sp = V.make_voice_splits(sel, cfg)
    assert audit_leaks(sel, sp) == {"group_leaks": [], "speaker_leaks": []}
    by = {r["idx"]: r for r in sel}
    for name in ("val", "test"):
        gs = [V.gender_of(by[i]) for i in sp[name] if by[i]["corpus"] == "cv_pt"]
        vs = {(V.voice_key(by[i]), V.gender_of(by[i])) for i in sp[name] if by[i]["corpus"] == "cv_pt"}
        nf = sum(1 for _, g in vs if g == "F")
        nm = sum(1 for _, g in vs if g == "M")
        assert nf == nm == 4, (name, nf, nm)
        assert gs
    tr = {(V.voice_key(by[i]), V.gender_of(by[i])) for i in sp["train"] if by[i]["corpus"] == "cv_pt"}
    assert sum(1 for _, g in tr if g == "F") == sum(1 for _, g in tr if g == "M") == 12


def test_balance_off_keeps_all():
    cfg = V.VoiceCfg(balance_gender=())
    sel = V.select_voices(_cv(), cfg)
    assert len({V.voice_key(r) for r in sel if r["corpus"] == "cv_pt"}) == 10 + 25 + 6
