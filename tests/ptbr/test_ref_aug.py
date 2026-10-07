import numpy as np

import ref_aug as A

SR = A.SR


def _speechlike(n=SR * 5, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    f0 = 140 + 30 * np.sin(2 * np.pi * 0.7 * t)
    ph = 2 * np.pi * np.cumsum(f0) / SR
    x = sum(np.sin(k * ph) / k for k in range(1, 20)) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    x = x + 0.01 * rng.standard_normal(n)
    return (x / np.max(np.abs(x)) * 0.8).astype(np.float32)


def _band_energy(x, lo, hi):
    s = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return float(s[(f >= lo) & (f < hi)].sum())


def test_length_dtype_finite_peak_over_many_seeds():
    x = _speechlike()
    for k in range(40):
        y, p = A.augment(x, A.rng_for(1, f"clip{k}", 0))
        assert y.shape == x.shape and y.dtype == np.float32
        assert np.all(np.isfinite(y)) and np.max(np.abs(y)) <= 1.0 + 1e-6
        assert p                                     # sempre aplica >= 1 estagio


def test_deterministic_and_variants_differ():
    x = _speechlike()
    a1, _ = A.augment(x, A.rng_for(1, "c", 0))
    a2, _ = A.augment(x, A.rng_for(1, "c", 0))
    b, _ = A.augment(x, A.rng_for(1, "c", 1))
    assert np.array_equal(a1, a2) and not np.array_equal(a1, b)


def test_noise_snr_is_accurate():
    x = _speechlike()
    rng = np.random.default_rng(0)
    for snr in (10.0, 20.0):
        y = A.add_noise(x.astype(np.float64), snr, "pink", rng)
        n = y - x
        meas = 20 * np.log10(A._rms(x) / A._rms(n))
        assert abs(meas - snr) < 0.01


def test_eq_changes_spectrum_and_keeps_rms():
    x = _speechlike().astype(np.float64)
    changed = 0
    for k in range(30):
        y, p = A.apply_eq(x, np.random.default_rng(k))
        assert abs(A._rms(y) - A._rms(x)) < 1e-6
        if "lowpass_hz" in p and p["lowpass_hz"] < 5000:
            assert _band_energy(y, 8000, 12000) < 0.5 * _band_energy(x, 8000, 12000)
            changed += 1
    assert changed > 0


def test_reverb_keeps_length_and_rms_and_smears():
    x = _speechlike().astype(np.float64)
    y, p = A.apply_reverb(x, np.random.default_rng(3))
    assert len(y) == len(x) and abs(A._rms(y) - A._rms(x)) < 1e-6
    assert not np.allclose(x, y) and 0.15 <= p["rt60"] <= 0.75


def test_mulaw_roundtrip_close_at_8bit_and_quantised():
    x = _speechlike()
    y = A.mulaw_roundtrip(x, bits=8)
    assert np.max(np.abs(y - x)) < 0.05
    assert len(np.unique(np.round(y, 6))) < len(np.unique(np.round(x, 6)))


def test_silence_and_short_inputs_do_not_crash():
    y, p = A.augment(np.zeros(SR, dtype=np.float32), np.random.default_rng(0))
    assert np.all(y == 0) and "skipped" in p
    y2, _ = A.augment(np.ones(10, dtype=np.float32) * 0.5, np.random.default_rng(0))
    assert len(y2) == 10


def test_choose_aug_variant_probability_and_determinism():
    import ref_aug as RA

    avail = [0, 1]
    got = [RA.choose_aug_variant(1, s, f"clip{s % 50}", avail, 0.5) for s in range(4000)]
    frac = sum(g is not None for g in got) / len(got)
    assert 0.45 < frac < 0.55
    assert {g for g in got if g is not None} == {0, 1}
    assert got == [RA.choose_aug_variant(1, s, f"clip{s % 50}", avail, 0.5) for s in range(4000)]
    assert RA.choose_aug_variant(1, 3, "a", None, 0.5) is None          # sem variantes
    assert RA.choose_aug_variant(1, 3, "a", [], 0.5) is None
    assert RA.choose_aug_variant(1, 3, "a", avail, 0.0) is None          # desligado
    assert all(RA.choose_aug_variant(1, s, "a", [3], 1.0) == 3 for s in range(20))
