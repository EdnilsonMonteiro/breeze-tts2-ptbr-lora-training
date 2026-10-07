import pytest

from asr_metrics import (bootstrap_ci, catastrophic, edit_ops, micro_wer, repetition_ratio,
                         summarize, wer_detail)


def test_edit_ops_counts():
    assert edit_ops("a b c".split(), "a x c".split()) == (1, 0, 0)
    assert edit_ops("a b c".split(), "a c".split()) == (0, 1, 0)
    assert edit_ops("a c".split(), "a b c".split()) == (0, 0, 1)


def test_wer_normalizes_numbers_both_sides():
    pytest.importorskip("num2words")
    d = wer_detail("Ele pagou R$ 15 na TV", "ele pagou quinze reais na tê vê")
    assert d["wer"] == 0.0


def test_wer_english_skips_pt_normalization():
    d = wer_detail("The quick brown fox.", "the quick brown fox", lang="en")
    assert d["wer"] == 0.0


def test_insertions_reported():
    d = wer_detail("ele foi para casa", "ele foi para casa casa casa")
    assert d["I"] == 2 and d["S"] == 0 and d["D"] == 0


def test_catastrophic():
    assert catastrophic(0.1, 4.0, 10, "ok") == []
    assert "wer" in catastrophic(0.9, 4.0, 10)
    assert "longo" in catastrophic(0.1, 30.0, 5)
    assert "repeticao" in catastrophic(0.1, 4.0, 12, "a b c a b c a b c a b c a b c")


def test_bootstrap_and_summary():
    m, lo, hi = bootstrap_ci([0.1] * 20)
    assert m == pytest.approx(0.1) and lo == pytest.approx(0.1) and hi == pytest.approx(0.1)
    s = summarize([0.0, 0.2, 0.4], [False, False, True])
    assert s["n"] == 3 and s["fail_rate"] == pytest.approx(1 / 3)
    assert summarize([])["n"] == 0
    assert micro_wer([{"S": 1, "D": 0, "I": 0, "n_ref": 4}, {"S": 0, "D": 0, "I": 0, "n_ref": 4}]) == 0.125
    assert repetition_ratio("a b c d e f") == 0.0
