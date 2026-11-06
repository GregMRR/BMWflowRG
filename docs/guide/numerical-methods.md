# Numerical methods

| Step | Method | Selectable |
|---|---|---|
| Integration in RG time | explicit Euler with fixed step `ds`, from `s = 0` down to `s_fin`; the dimensionful parameters `X` use the same step (`X += ds * eta * X`) | step `ds` only (Crank–Nicolson is planned, see [status](../STATUS.md)) |
| Approximation of the flow equations | NLO (`f(p, w)`) or LO (`f(p)`); `f_lambda = 1` in all dimensions | `approximation` |
| Loop integrals over `q`, `theta` | Gauss–Legendre quadrature on `[0, q_max]` × `[0, theta_max]` (`flow/models/maths_utils.py`); trivial angular grid in 1D; the internal frequency integral is done analytically | `deg_q`, `q_max`, `deg_theta`, `theta_max` |
| Anomalous dimensions | the 2×2 linear system for `eta_D`, `eta_nu` is solved in closed form at each step | – |
| Functions between grid points | cubic splines in `p` (`scipy.interpolate.CubicSpline`), extended beyond `p_max` (up to `p_max + q_max`) by a power law fitted on the last 5 points; in NLO, also cubic splines in `w` | grid sizes only |
| Derivatives `p ∂f/∂p`, `w ∂f/∂w` | derivatives of those splines | – |
| `I_nu` at `p = 0` | constant or quadratic extrapolation from `p_1`, `p_2` | `Integral_extrapol` |
| Exit to the dimensionful grid | point `p_two[i]` exits at `kappa = p_two[i]/p_exit`; `f(w)` at `p_exit` is taken from the `w` spline, and beyond `w_max` it is continued by a power law (mean slope of the last 5 `w` points) | `p_exit`, `p_two`/`w_two` grids |
| `I_dimful` (evo2, `'full'`) | `f(q, omega)` from a bicubic `RectBivariateSpline`, then Gauss–Legendre over `q`, `theta`, `omega` | `step_two_action`, `deg_omega`, `omega_max` |

The integrals are vectorised with NumPy. Numba and JAX versions of the KPZ kernels (`kpz_numbafunc.py`, `kpz_jaxfunc.py`, `model_kpz_jit.py`, `maths_utils_numba.py`, `maths_utils_jax.py`) are in `flow/models/` but are not used: `flow/models/__init__.py` imports the NumPy `ModelKPZ`.

