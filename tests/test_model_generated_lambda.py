"""Contract v3 (FRGscalarMSR docs/handoff/BMWflow_LO.md): the generated KPZ LO module with a flowing f_lambda
(ETAS = ("etaD", "etanu", None), loop at fp). At f_lambda = 1 its fD, fnu rows equal the KPZ module's; f_lambda(0) = 1
is protected (no eta) and must stay 1 along the flow."""
import numpy as np
import pytest

from flow import create, flow_dataclasses
from flow.evolution import Evolution
from flow.models import ModelGenerated
from flow.models.model_generated import load_generated
from test_model_generated import GRID_EXT, GRID_INT, TOL, prepare, r, r_, rel, state


def lambda_state(model):
    """The KPZ test state with f_lambda = 1 appended."""
    f, g, eta = state(model)
    return np.concatenate([f, np.ones((1, *f.shape[1:]))]), g, eta


def test_contract_accepted(lambda_path):
    gen = ModelGenerated(lambda_path, "LO", 2, GRID_EXT, GRID_INT, r, r_)
    assert gen.n_f == 3 and gen.n_eta == 2 and gen.eta_names == ("etaD", "etanu")


@pytest.mark.parametrize("attr, value", [
    ("CONTRACT_VERSION", 2),
    ("ETAS", ("etaD", "etanu")),          # wrong length
    ("ETAS", ("etaD", "etanu", 0)),       # neither a name nor None
    ("ETAS", ("etaD", "etaD", None)),     # duplicate name
])
def test_contract_refused(lambda_path, attr, value):
    mod = load_generated(lambda_path)
    setattr(mod, attr, value)
    with pytest.raises(ValueError):
        ModelGenerated(mod, "LO", 2, GRID_EXT, GRID_INT, r, r_)


@pytest.mark.parametrize("name", ["None", "flam", "etaX"])
def test_fixed_etas_only_non_none(lambda_path, name):
    with pytest.raises(ValueError, match="unknown etas"):
        ModelGenerated(lambda_path, "LO", 2, GRID_EXT, GRID_INT, r, r_, fixed_etas={name: 0.0})


def test_fixed_eta_reproduces_full_solve(lambda_path):
    gen = ModelGenerated(lambda_path, "LO", 2, GRID_EXT, GRID_INT, r, r_)
    f, g, eta = lambda_state(gen)
    prepare(gen, f, g, eta)
    full = gen.eta_calc(g)
    red = ModelGenerated(lambda_path, "LO", 2, GRID_EXT, GRID_INT, r, r_, fixed_etas={"etanu": full[1]})
    prepare(red, f, g, eta)
    assert np.allclose(red.eta_calc(g), full, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("mode", ["const", "quad"])
def test_flambda_one_equals_kpz(generated_path, lambda_path, dim, mode):
    """At f_lambda = 1 the fD, fnu rows equal those of the KPZ module, at the KPZ acceptance tolerances."""
    SINGLE, _ = TOL[dim]
    kpz = ModelGenerated(generated_path, "LO", dim, GRID_EXT, GRID_INT, r, r_, mode)
    lam = ModelGenerated(lambda_path, "LO", dim, GRID_EXT, GRID_INT, r, r_, mode)
    f, g, eta = lambda_state(lam)
    prepare(kpz, f[:2], g, eta)
    prepare(lam, f, g, eta)

    assert np.all(np.isfinite(lam.Integral))
    assert np.all(rel(lam.Integral[:2], kpz.Integral) <= SINGLE), rel(lam.Integral[:2], kpz.Integral)
    p0 = np.max(np.abs(lam.Integral[:2, 0] - kpz.Integral[:, 0]), axis=1) / np.max(np.abs(kpz.Integral), axis=(1, 2))
    assert np.all(p0 <= SINGLE), p0

    eta_k, eta_l = kpz.eta_calc(g), lam.eta_calc(g)
    assert eta_l.shape == (2,)
    assert np.all(rel(eta_l[:, None], eta_k[:, None]) <= 1e-11), (eta_l, eta_k)
    assert abs(lam.g_rhs_calc(g, eta_k) - kpz.g_rhs_calc(g, eta_k)) <= 1e-11 * abs(kpz.g_rhs_calc(g, eta_k))
    rhs_l, rhs_k = lam.f_rhs_calc(eta_k, f), kpz.f_rhs_calc(eta_k, f[:2])
    assert np.all(rel(rhs_l[:2], rhs_k) <= SINGLE), rel(rhs_l[:2], rhs_k)
    assert np.all(np.isfinite(rhs_l))


@pytest.mark.parametrize("dim", [1, 2])
def test_evolution_keeps_flambda0(lambda_path, dim, tmp_path):
    """2000 Euler steps (ds = 1e-3): f_lambda(p = 0) stays 1 to rounding, everything finite."""
    gen = ModelGenerated(lambda_path, "LO", dim, GRID_EXT, GRID_INT, r, r_, "const")
    ic = flow_dataclasses.IC_NLO(g_in=1.0, etas_in=np.zeros(2), fs_in=np.ones((3, GRID_EXT.N_p, 1)))
    evo = Evolution(gen, ic, 1e-3, str(tmp_path / "run"))
    evo.rg_evolution(s_fin=-2.0 + 1e-9, n_print=10 ** 9, n_save_params=10 ** 9, n_save_f=10 ** 9)
    assert abs(evo.f[2, 0, 0] - 1) <= 1e-12, evo.f[2, 0, 0] - 1
    assert np.max(np.abs(evo.f[2] - 1)) > 1e-6  # f_lambda does flow away from p = 0
    assert np.all(np.isfinite(evo.f)) and np.all(np.isfinite(evo.eta)) and np.isfinite(evo.g)


def test_create_and_eta_shape(lambda_path, tmp_path):
    params = (3, "LO", 1, GRID_EXT, GRID_INT, r, r_)
    ic = flow_dataclasses.IC_NLO(g_in=1.0, etas_in=np.zeros(2), fs_in=np.ones((3, GRID_EXT.N_p, 1)))
    evo = create.create(lambda_path, params, "evo1", (ic, 1e-3, str(tmp_path / "run")))
    assert evo.model.n_eta == 2
    bad = flow_dataclasses.IC_NLO(g_in=1.0, etas_in=np.zeros(3), fs_in=np.ones((3, GRID_EXT.N_p, 1)))
    with pytest.raises(Exception, match="not consistent"):
        create.create(lambda_path, params, "evo1", (bad, 1e-3, str(tmp_path / "bad")))
