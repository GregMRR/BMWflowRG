# FRG

Package for Functional Renormalisation Group calculations within the next-to-leading order (NLO) approximation and the *two grids* scheme. 

## About the FRG

In a nutshell, the Renormalisation Group method is a smart coarse-graining: it tells how a system's effective description changes with scale.

The FRG method [1] is used and shows impressive results in various domains of physics: equilibrium and out-of-equilibrium phenomena, quantum and classical, from condensed matter to quantum gravity. Here, we focus on classical out-of-equilibrium physics: stochastic Navier-Stokes equation, Kardar-Parisi-Zhang equation, while the calculations for other models (whatever physics they describe) can be implemented in the same manner, which was the motivation of creating this repository.

Solution of the FRG equation - the Wetterich equation - is possible under some approximation. In this repository we implement the NLO approximation, which is very expressive and captures well the two-point correlations.

On top of that, we reinforce this method with the *two grids* scheme [2], which permits to obtain two-point correlation functions in a wide range of length- and time-scales.

## Install

The package is not on PyPI and has no installer: clone the repository and install the pinned dependencies.

```bash
git clone <repository-url> BMWflow
cd BMWflow
pip install -r flow/requirements.txt   # numpy, scipy, numba, matplotlib
pip install jupyter                    # only to run the notebooks in usage_examples/
```

To import `flow`, put the repository root on `sys.path` (the notebooks do this from `usage_examples/`):

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))   # path of the repository root
```

## Usage

### Quickstart: 1D KPZ, NLO, two grids

Condensed from `usage_examples/KPZ_1D_twogrids.ipynb`:

```python
import numpy as np
from flow import flow_dataclasses, regulator, create

# --- Model ---
n_f = 2                  # KPZ: two flowing functions f_D, f_nu (and eta_D, eta_nu)
approximation = "NLO"    # or "LO"
dim = 1

alpha, beta = 2, 1
r  = lambda x: regulator.wett_alpha_beta(x, alpha, beta)    # regulator r(q)
r_ = lambda x: regulator.wett_alpha_beta_(x, alpha, beta)   # its derivative

N_p, p_min, p_max = 100, 0.01, 100
N_w, w_min, w_max = 100, 0.01, 100
q_max, deg_q, deg_theta = 4, 16, 10

grid_ext = flow_dataclasses.ParamsGridExternal(N_p=N_p, p_min=p_min, p_max=p_max,
                                               N_w=N_w, w_min=w_min, w_max=w_max)
grid_int = flow_dataclasses.ParamsGridInternal(deg_q=deg_q, q_max=q_max, deg_theta=deg_theta)
model_params = (n_f, approximation, dim, grid_ext, grid_int, r, r_)

# --- Dimensionless evolution ---
IC = flow_dataclasses.IC_NLO(g_in=1.0,
                             etas_in=np.zeros(n_f),
                             fs_in=np.ones((n_f, N_p, N_w)))
ds = 0.001
evo_params = (IC, ds, "KPZ_1D_twogrids")   # results are written to this directory

# --- Dimensionful (second) grid ---
evo_two_params = dict(
    X_dimful_in=np.array([1., 1.]),        # [D, nu] at kappa = Lambda
    match_w_to_p_two=True,                 # w_two = p_two**2
    p_exit=q_max,
    Np_two=200, p_min_two=0.01, p_max_two=q_max,
    Nw_two=200, w_min_two=None, w_max_two=None,
    scale="log",
    s_fin_all_exited=True,                 # stop once every p_two point has exited
    step_two_action="full", deg_omega=500, omega_max=w_max,
)

flow = create.create("KPZ", model_params, "evo2", evo_params, evo_two_params)
flow.rg_evolution(s_fin=-20, n_print=1000, n_save_params=100, n_save_f=1000)
```

For the dimensionless flow alone, use `"evo1"` and leave out the last argument:

```python
flow = create.create("KPZ", model_params, "evo1", evo_params)
flow.rg_evolution(s_fin=-20, n_print=1000, n_save_params=100, n_save_f=1000)
```

Notes:
- RG time is `s = ln(kappa/Lambda)`, and it runs from `0` down to `s_fin`, so `s_fin < 0` and `ds > 0`.
- The output directory is created. If it already exists, `Evolution` asks on stdin `Path exists. Press y to continue` and exits on any other key, so a non-interactive run must write to a new directory.

## Documentation

All documentation is in [docs/](docs/README.md):

- Guides: [code layout and the two-grids scheme](docs/guide/overview.md), [reading the results](docs/guide/results.md),
  [numerical methods](docs/guide/numerical-methods.md), [generated LO models](docs/guide/generated-models.md),
  [adding a model](docs/guide/adding-a-model.md), [development](docs/guide/development.md).
- [API reference](docs/reference/README.md), generated from the docstrings in `flow/`: every parameter of `create`, the
  models, the grids, the initial condition, the regulators and the evolutions.
- [Status](docs/STATUS.md): what is implemented and planned.

---------

#### Author
Liubov Gosteva

#### Licence
GNU General Public License v3.0: see [LICENSE.md](LICENSE.md).

#### References
[1] N. Dupuis, L. Canet, A. Eichhorn, W. Metzner, J.M. Pawlowski, M. Tissier, N. Wschebor, The nonperturbative functional renormalization group and its applications,
Physics Reports, V.910, p. 1-114 (2021), https://doi.org/10.1016/j.physrep.2021.01.001.

[2] L.Gosteva, N. Wschebor, L. Canet, Unveiling the different scaling regimes of the one-dimensional Kardar–Parisi–Zhang–Burgers equation using the functional renormalisation group, J. Stat. Mech., p. 114002 (2025), https://doi.org/10.1088/1742-5468/ae1b85.