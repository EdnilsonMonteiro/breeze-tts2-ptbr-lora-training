import pytest

from refs import (DEFAULT_MIX, build_pools, eligible_ref_speaker, mix_from_ref_edit_frac,
                  parse_mix, pick_ref, resolve_condition, restrict_to_ref, sample_variant)


def _recs():
    r = []
    for c in range(6):
        r.append({"idx": f"a{c}", "speaker": "A", "dur_proc_s": 5.0})
    r.append({"idx": "b0", "speaker": "B", "dur_proc_s": 5.0})              # locutor de 1 clipe
    for c in range(4):
        r.append({"idx": f"u{c}", "speaker": "show:99", "dur_proc_s": 5.0})  # bucket dissolvido
    r.append({"idx": "long", "speaker": "A", "dur_proc_s": 12.0})           # longo demais p/ ref
    return r


def test_pools_exclude_unassigned_and_long():
    pools = build_pools(_recs(), [r["idx"] for r in _recs()])
    assert "show:99" not in pools and "long" not in pools["A"]
    assert not eligible_ref_speaker("x:99") and eligible_ref_speaker("x:3")


def test_never_self_and_same_speaker():
    recs = _recs()
    pools = build_pools(recs, [r["idx"] for r in recs])
    for r in recs:
        for ep in range(5):
            ref = pick_ref(r, pools, ep)
            if r["speaker"] == "A":
                assert ref is not None and ref != r["idx"] and ref.startswith("a")
    assert pick_ref(recs[6], pools) is None              # B: so 1 clipe
    assert pick_ref(recs[7], pools) is None              # :99


def test_ref_varies_by_epoch_and_is_deterministic():
    recs = _recs()
    pools = build_pools(recs, [r["idx"] for r in recs])
    a0 = recs[0]
    seq = [pick_ref(a0, pools, ep) for ep in range(30)]
    assert len(set(seq)) > 1
    assert seq == [pick_ref(a0, pools, ep) for ep in range(30)]


def test_pool_is_split_local():
    recs = _recs()
    pools = build_pools(recs, ["a0", "a1"])
    assert pick_ref(recs[0], pools) == "a1"


def test_resolve_condition_fallback_and_require_ref():
    recs = _recs()
    pools = build_pools(recs, [r["idx"] for r in recs])
    mix = {"ref_edit_tata": 1.0}
    assert resolve_condition(recs[7], pools, mix)[0] == "tts_instruction"      # sem ref -> cai
    assert resolve_condition(recs[7], pools, mix, require_ref=True) is None
    v, ref = resolve_condition(recs[0], pools, mix)
    assert v == "ref_edit_tata" and ref != "a0"


def test_mix_distribution_and_parsing():
    assert sum(DEFAULT_MIX.values()) == pytest.approx(1.0)
    n = 4000
    got = {}
    for i in range(n):
        v = sample_variant(f"i{i}", DEFAULT_MIX)
        got[v] = got.get(v, 0) + 1
    assert got["ref_edit_tata"] / n == pytest.approx(0.75, abs=0.03)
    assert parse_mix("ref_edit=3,plain=1") == {"ref_edit_tata": 0.75, "tts_plain": 0.25}
    assert restrict_to_ref(DEFAULT_MIX).keys() == {"ref_edit_tata", "ref_clone_tata"}
    assert mix_from_ref_edit_frac(1.0) == {"ref_edit_tata": 1.0}
    with pytest.raises(ValueError):
        parse_mix("foo=1")
