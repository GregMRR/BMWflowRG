"""splines_pplusq: power-law tail beyond p_max, constant when f(p_max) = 0."""
import numpy as np

from flow.models.grids import grid_log0
from flow.models.splines import splines_pplusq

P = grid_log0(1e-2, 1e2, 100)
Q = np.linspace(0.1, 4, 16)
EXT = (P[-1] + Q, (P[-1] + Q) / P[-1])


def test_zero_function_stays_zero():
    spl = splines_pplusq(np.zeros((P.size, 1)), 1, P, *EXT)
    x = np.concatenate([P, EXT[0]])
    assert np.all(spl[0](x) == 0)


def test_power_law_tail_unchanged():
    f = (1 + P ** 2)[:, None] ** -0.7
    spl = splines_pplusq(f, 1, P, *EXT)
    b = P[-1] * np.gradient(f[-5:, 0], P[-5:])[-1] / f[-1, 0]
    assert np.allclose(spl[0](EXT[0]), f[-1, 0] * EXT[1] ** b, rtol=1e-12)


def test_tail_through_zero_is_constant():
    """A tail that changes sign (f ~ 0 at p_max) gets no power law: tiny f_m with large f' would give a huge b."""
    f = np.zeros((P.size, 1))
    f[-5:, 0] = [1e-20, -3e-21, 2e-22, -1e-23, 1e-28]
    spl = splines_pplusq(f, 1, P, *EXT)
    assert np.allclose(spl[0](EXT[0]), 1e-28, rtol=1e-12)
