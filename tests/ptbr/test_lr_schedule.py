import pytest

from lr_schedule import cosine_lr_factor as f


def test_warmup_and_peak():
    assert f(0, 1000, 50, 0.0, 1) == 0.0
    assert f(25, 1000, 50, 0.0, 1) == pytest.approx(0.5)
    assert f(50, 1000, 50, 0.0, 1) == pytest.approx(1.0)


def test_single_cosine_ends_near_zero():
    assert f(999, 1000, 50, 0.0, 1) < 0.01
    assert f(1000, 1000, 50, 0.0, 1) == pytest.approx(0.0, abs=1e-9)


def test_restarts_have_floor_and_no_step_at_cycle_end():
    total, w, floor, cyc = 3050, 50, 0.15, 3
    vals = [f(s, total, w, floor, cyc) for s in range(w, total)]
    assert min(vals) >= floor - 1e-9 and max(vals) <= 1.0 + 1e-9
    cycle = (total - w) // cyc
    # ultimo passo de um ciclo ~ piso; primeiro do proximo = pico (restart), sem degrau antes
    assert f(w + cycle - 1, total, w, floor, cyc) == pytest.approx(floor, abs=0.01)
    assert f(w + cycle, total, w, floor, cyc) == pytest.approx(1.0)
    # monotonicamente decrescente dentro do ciclo (a versao antiga tinha platô no fim)
    inner = [f(s, total, w, floor, cyc) for s in range(w, w + cycle)]
    assert all(a >= b - 1e-12 for a, b in zip(inner, inner[1:]))
