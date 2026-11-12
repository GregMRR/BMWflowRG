"""Contract v4, NLO kind (FRGscalarMSR docs/handoff/BMWflow_LO.md, design/NLOPIPELINE.md): generated modules whose functions
depend on (p, w). KPZ_NLO.py (f_lambda = 1) must equal the hand-written ModelKPZ in NLO; KPZLambda_NLO.py (flowing f_lambda)
must equal it at f_lambda = 1, equal the LO module on the w = 0 column, and match Mathematica at reference points.
Fixtures: FRGscalarMSR benchmarks/KPZNLO/export.wl."""
import json
from pathlib import Path

import numpy as np
import pytest

from flow import flow_dataclasses
from flow.evolution import Evolution
from flow.models import ModelGenerated, ModelKPZ
from flow.models.model_generated import load_generated
from test_model_generated import GRID_INT, r, r_, rel

FIX = Path(__file__).parent / "fixtures"
KPZ_NLO, LAM_NLO, LAM_LO = FIX / "KPZ_NLO.py", FIX / "KPZLambda_NLO.py", FIX / "KPZLambda_LO.py"
GRID = flow_dataclasses.ParamsGridExternal(N_p=40, p_min=0.01, p_max=100, N_w=12, w_min=0.01, w_max=100)
TOL = 1e-10  # max_grid |delta| / max_grid |x| per function (two algebraic forms of the same rows; design/NLOPIPELINE.md review E8)


def nlo_state(model, n_f):
    """A non-trivial NLO state, shape (n_f, Np, Nw), with f(0, w) = 1 for f_nu and f_lambda as the Ward identities require."""
    p2 = model.p[None, :, None] ** 2
    w2 = model.w[None, None, :] ** 2
    f = [1 + 0.4 * p2 / (1 + p2) / (1 + w2), 1 - 0.3 * p2 / (2 + p2) * (1 + 0.1 * w2 / (1 + w2))]
    if n_f == 3:
        f.append(1 + 0.2 * p2 / (3 + p2) / (1 + 0.5 * w2))
    return np.concatenate(f), 1.3, np.array([0.1, -0.2])


def prepare(model, f, g, eta):
    model.f_spline_upd(f)
    model.calc_upd()
    model.Integral_upd(g, eta)


def test_contract():
    assert ModelGenerated(LAM_NLO, "NLO", 2, GRID, GRID_INT, r, r_).n_f == 3
    with pytest.raises(ValueError):
        ModelGenerated(LAM_NLO, "LO", 2, GRID, GRID_INT, r, r_)  # an NLO module requested as LO
    with pytest.raises(ValueError):
        ModelGenerated(LAM_LO, "NLO", 2, GRID, GRID_INT, r, r_)  # an LO module requested as NLO
    lo = load_generated(LAM_LO)
    lo.CONTRACT_VERSION = 3  # v3 LO modules stay readable
    assert ModelGenerated(lo, "LO", 2, GRID, GRID_INT, r, r_).n_f == 3
    nlo = load_generated(LAM_NLO)
    nlo.CONTRACT_VERSION = 3  # v3 has no NLO kind
    with pytest.raises(ValueError):
        ModelGenerated(nlo, "NLO", 2, GRID, GRID_INT, r, r_)


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("mode", ["const", "quad"])
@pytest.mark.parametrize("path", [KPZ_NLO, LAM_NLO])
def test_equals_kpz_nlo(path, dim, mode):
    """f_lambda = 1: Integral, eta, beta and f_rhs (with (2 - eta_nu) w df/dw) equal ModelKPZ NLO."""
    kpz = ModelKPZ(2, "NLO", dim, GRID, GRID_INT, r, r_, mode)
    gen = ModelGenerated(path, "NLO", dim, GRID, GRID_INT, r, r_, mode)
    f, g, eta = nlo_state(kpz, 2)
    fg = np.concatenate([f, np.ones((gen.n_f - 2, *f.shape[1:]))])
    prepare(kpz, f, g, eta)
    prepare(gen, fg, g, eta)
    assert np.all(np.isfinite(gen.Integral))
    assert np.all(rel(gen.Integral[:2], kpz.Integral) <= TOL), rel(gen.Integral[:2], kpz.Integral)
    eta_k = kpz.eta_calc(g)
    assert np.all(rel(gen.eta_calc(g)[:, None], eta_k[:, None]) <= TOL)
    assert abs(gen.g_rhs_calc(g, eta_k) - kpz.g_rhs_calc(g, eta_k)) <= TOL * abs(kpz.g_rhs_calc(g, eta_k))
    assert np.all(rel(gen.f_rhs_calc(eta_k, fg)[:2], kpz.f_rhs_calc(eta_k, f)) <= TOL)


@pytest.mark.parametrize("dim", [1, 2])
def test_w0_column_equals_lo(dim):
    """The w = 0 column of the NLO flow is the LO flow (flowing f_lambda), at the same f(p, 0)."""
    nlo = ModelGenerated(LAM_NLO, "NLO", dim, GRID, GRID_INT, r, r_, "quad")
    lo = ModelGenerated(LAM_LO, "LO", dim, GRID, GRID_INT, r, r_, "quad")
    f, g, eta = nlo_state(nlo, 3)
    prepare(nlo, f, g, eta)
    prepare(lo, f[:, :, :1], g, eta)
    assert np.all(rel(nlo.Integral[:, :, 0], lo.Integral[:, :, 0]) <= TOL), rel(nlo.Integral[:, :, 0], lo.Integral[:, :, 0])
    eta_n = nlo.eta_calc(g)
    assert np.allclose(eta_n, lo.eta_calc(g), rtol=TOL, atol=0)
    assert np.all(rel(nlo.f_rhs_calc(eta_n, f)[:, :, 0], lo.f_rhs_calc(eta_n, f[:, :, :1])[:, :, 0]) <= TOL)


@pytest.mark.parametrize("dim", [1, 2])
def test_flambda_ward_every_w(dim):
    """KCW Eq. 39: the f_lambda loop row vanishes at p = 0 for every w (f_lambda(w, 0) stays 1)."""
    gen = ModelGenerated(LAM_NLO, "NLO", dim, GRID, GRID_INT, r, r_)
    f, g, eta = nlo_state(gen, 3)
    prepare(gen, f, g, eta)
    assert np.max(np.abs(gen.Integral[2, 0, :])) <= 1e-12 * np.max(np.abs(gen.Integral[2]))
    assert np.max(np.abs(gen.Integral[2, 1:, 1:])) > 0  # and the row is not zero elsewhere


def test_reference_points():
    """Kernels against Mathematica (computed with 30 digits) at the points of FRGscalarMSR benchmarks/KPZNLO/export.wl, two at w = 0
    where the LO module must agree too."""
    nlo, lo = load_generated(LAM_NLO), load_generated(LAM_LO)
    for pt in json.loads((FIX / "KPZLambda_NLO_points.json").read_text()):
        args = [pt[k] for k in ("q", "c", "Q2", "fp", "fq", "fQ", "rq", "rq_", "rQ", "eta", "couplings", "d")]
        val = np.array(nlo.loop(pt["p"], pt["w"], *args), dtype=float)
        assert np.allclose(val, pt["loop"], rtol=TOL, atol=0), (pt["w"], val, pt["loop"])  # 2e-11 at w = 3.375: a cancelling f_D value
        if pt["loopLO"] is not None:
            assert np.allclose(np.array(lo.loop(pt["p"], *args), dtype=float), pt["loopLO"], rtol=TOL, atol=0)


def test_evolution_kpz_nlo(tmp_path):
    """A short NLO run (d = 1, 300 steps): the generated KPZ module follows ModelKPZ."""
    kpz = ModelKPZ(2, "NLO", 1, GRID, GRID_INT, r, r_, "const")
    gen = ModelGenerated(KPZ_NLO, "NLO", 1, GRID, GRID_INT, r, r_, "const")
    ic = lambda: flow_dataclasses.IC_NLO(g_in=1.0, etas_in=np.zeros(2), fs_in=np.ones((2, GRID.N_p, GRID.N_w)))
    ek = Evolution(kpz, ic(), 1e-3, str(tmp_path / "kpz"))
    eg = Evolution(gen, ic(), 1e-3, str(tmp_path / "gen"))
    for e in (ek, eg):
        e.rg_evolution(s_fin=-0.3 + 1e-9, n_print=10 ** 9, n_save_params=10 ** 9, n_save_f=10 ** 9)
    assert np.all(np.isfinite(eg.f))
    assert np.all(rel(eg.f, ek.f) <= TOL), rel(eg.f, ek.f)
    assert np.all(rel(eg.eta[:, None], ek.eta[:, None]) <= TOL)
    assert abs(eg.g - ek.g) <= TOL * abs(ek.g)
