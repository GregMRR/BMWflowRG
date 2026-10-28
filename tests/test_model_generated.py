"""Acceptance of ModelGenerated (FRGscalarMSR docs/handoff/BMWflow_LO.md): the generated KPZ LO module
against the hand-written ModelKPZ in LO, d = 1 and 2, on the grids and regulator of
usage_examples/KPZ_1D_twogrids."""
import numpy as np
import pytest

from flow import create, flow_dataclasses, regulator
from flow.evolution import Evolution
from flow.models import ModelGenerated, ModelKPZ

ALPHA, BETA = 2, 1
r = lambda x: regulator.wett_alpha_beta(x, ALPHA, BETA)
r_ = lambda x: regulator.wett_alpha_beta_(x, ALPHA, BETA)
GRID_EXT = flow_dataclasses.ParamsGridExternal(N_p=100, p_min=0.01, p_max=100, N_w=100, w_min=0.01, w_max=100)
GRID_INT = flow_dataclasses.ParamsGridInternal(deg_q=16, q_max=4, deg_theta=10)
N_F = 2

# Handoff tolerances (max_grid |delta| / max_grid |x| per function): 1e-12 for one evaluation, 1e-11 after the run.
# d = 1 is looser: the p-grid point 0.02121 lies 1.1e-5 from the q node 0.02120, so at theta = pi the loop is
# evaluated at Q ~ 1e-5 where r(Q) ~ 1e10. There the integrand is ill-conditioned: ModelKPZ itself is 1e-8 away
# from a long-double evaluation, and two arrangements of the same formula differ by ~1e-11 (measured 8e-12, 2.2e-11).
TOL = {1: (2e-11, 1e-10), 2: (1e-12, 1e-11)}


def rel(a, b):
    """Per function: max_grid |a - b| / max_grid |b| (the handoff's norm)."""
    a, b = np.atleast_2d(np.asarray(a, dtype=float)), np.atleast_2d(np.asarray(b, dtype=float))
    a, b = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    return np.max(np.abs(a - b), axis=1) / np.max(np.abs(b), axis=1)


def models(path, dim, mode):
    kpz = ModelKPZ(N_F, "LO", dim, GRID_EXT, GRID_INT, r, r_, mode)
    gen = ModelGenerated(path, "LO", dim, GRID_EXT, GRID_INT, r, r_, mode)
    return kpz, gen


def state(model):
    """A non-trivial LO state with f(0) = 1."""
    p2 = model.p[:, None] ** 2
    f = np.array([1 + 0.4 * p2 / (1 + p2), 1 - 0.3 * p2 / (2 + p2)])
    return f, 1.3, np.array([0.1, -0.2])


def prepare(model, f, g, eta):
    model.f_spline_upd(f)
    model.calc_upd()
    model.Integral_upd(g, eta)


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("mode", ["const", "quad"])
def test_single_evaluation(generated_path, dim, mode):
    SINGLE, RUN = TOL[dim]
    kpz, gen = models(generated_path, dim, mode)
    f, g, eta = state(kpz)
    prepare(kpz, f, g, eta)
    prepare(gen, f, g, eta)

    assert np.all(np.isfinite(gen.Integral))
    assert np.all(rel(gen.Integral, kpz.Integral) <= SINGLE), rel(gen.Integral, kpz.Integral)
    p0 = np.max(np.abs(gen.Integral[:, 0] - kpz.Integral[:, 0]), axis=1) / np.max(np.abs(kpz.Integral), axis=(1, 2))
    assert np.all(p0 <= SINGLE), p0  # the p = 0 row (extrapolated for P2_DIVIDED rows)

    eta_k, eta_g = kpz.eta_calc(g), gen.eta_calc(g)
    assert np.all(rel(eta_g[:, None], eta_k[:, None]) <= 1e-11), (eta_g, eta_k)
    assert abs(gen.g_rhs_calc(g, eta_k) - kpz.g_rhs_calc(g, eta_k)) <= 1e-11 * abs(kpz.g_rhs_calc(g, eta_k))
    assert np.all(rel(gen.f_rhs_calc(eta_k, f), kpz.f_rhs_calc(eta_k, f)) <= SINGLE)


@pytest.mark.parametrize("dim", [1, 2])
def test_evolution(generated_path, dim, tmp_path):
    SINGLE, RUN = TOL[dim]
    kpz, gen = models(generated_path, dim, "const")
    ic = lambda: flow_dataclasses.IC_NLO(g_in=1.0, etas_in=np.zeros(N_F), fs_in=np.ones((N_F, GRID_EXT.N_p, 1)))
    ek = Evolution(kpz, ic(), 1e-3, str(tmp_path / "kpz"))
    eg = Evolution(gen, ic(), 1e-3, str(tmp_path / "gen"))
    for e in (ek, eg):
        e.rg_evolution(s_fin=-2.0 + 1e-9, n_print=10 ** 9, n_save_params=10 ** 9, n_save_f=10 ** 9)
    assert np.all(rel(eg.f, ek.f) <= RUN), rel(eg.f, ek.f)
    assert np.all(rel(eg.eta[:, None], ek.eta[:, None]) <= RUN), (eg.eta, ek.eta)
    assert abs(eg.g - ek.g) <= RUN * abs(ek.g)
    assert np.all(np.isfinite(eg.f))


def test_create_accepts_a_generated_module(generated_path, tmp_path):
    params = (N_F, "LO", 1, GRID_EXT, GRID_INT, r, r_)
    ic = flow_dataclasses.IC_NLO(g_in=1.0, etas_in=np.zeros(N_F), fs_in=np.ones((N_F, GRID_EXT.N_p, 1)))
    evo = create.create(generated_path, params, "evo1", (ic, 1e-3, str(tmp_path / "run")))
    assert isinstance(evo.model, ModelGenerated)
    with pytest.raises(ValueError):
        create.create(generated_path, (3, *params[1:]), "evo1", (ic, 1e-3, str(tmp_path / "bad")))
    with pytest.raises(ValueError):
        create.create(generated_path, params, "evo2", (ic, 1e-3, str(tmp_path / "bad2")), {})


def test_contract_checks(generated_path):
    from flow.models.model_generated import load_generated
    mod = load_generated(generated_path)
    for attr, value, args in [("CONTRACT_VERSION", 1, ("LO", 1)), ("COUPLINGS", ("alamk",), ("LO", 1)),
                              ("FIXED_DIMENSION", 2, ("LO", 1))]:
        old = getattr(mod, attr)
        setattr(mod, attr, value)
        try:
            with pytest.raises(ValueError):
                ModelGenerated(mod, *args[:1], args[1], GRID_EXT, GRID_INT, r, r_)
        finally:
            setattr(mod, attr, old)
    with pytest.raises(ValueError):
        ModelGenerated(mod, "NLO", 1, GRID_EXT, GRID_INT, r, r_)
    with pytest.raises(ValueError):
        ModelGenerated(mod, "LO", 1, GRID_EXT, GRID_INT, r, r_, regulators={"rx": (r, r_)})


@pytest.mark.parametrize("fixed", [0, 1])
def test_fixed_etas_reproduce_full_solve(generated_path, fixed):
    """Fixing one eta at the value of the full solve returns the full-solve vector."""
    _, gen = models(generated_path, 2, "const")
    f, g, eta = state(gen)
    prepare(gen, f, g, eta)
    full = gen.eta_calc(g)
    name = gen.module.ETAS[fixed]
    red = ModelGenerated(generated_path, "LO", 2, GRID_EXT, GRID_INT, r, r_, "const", None, {name: full[fixed]})
    prepare(red, f, g, eta)
    assert np.allclose(red.eta_calc(g), full, rtol=1e-12, atol=1e-14), (red.eta_calc(g), full)


def test_fixed_etas_unknown_name(generated_path):
    with pytest.raises(ValueError, match="unknown etas"):
        ModelGenerated(generated_path, "LO", 2, GRID_EXT, GRID_INT, r, r_, fixed_etas={"etaX": 1.0})
