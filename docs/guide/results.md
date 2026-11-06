# Reading the results

`flow.file_utils.read_par_bin(path, file, shape)` reads the binary files as an array of shape `(n_steps, *shape)`:

| File | Written | Shape per record | Content |
|---|---|---|---|
| `f.bin` | every `n_save_f` steps | `(n_f, N_p, N_w)`, same as `fs_in` | dimensionless functions `f` on the `(p, w)` grid |
| `flow_parameters.bin` | every `n_save_params` steps | `(1 + n_f + 1,)` | `[s, eta_1..eta_nf, g]` |
| `exit_parameters.bin` | at each exit (evo2) | `(2 + n_f,)` | `[js_exit, s, X_dimful...]` (for KPZ `X_dimful = [D, nu]`) |
| `G20_IC_two.npy` | overwritten at each exit (evo2) | `(Np_two, Nw_two)` | dimensionful correlation function `G(p_two, w_two)`, the initial condition for the large-p equation |
| `I_dimful.bin` | every step (evo2, `step_two_action="full"`) | `(Np_two,)` | diffusion coefficient for the right-hand side of the large-p equation |
| `I_inner.dat` | every step (evo2, `"full"`) | text, one value per line | the p-independent part of `I_dimful` |

```python
from flow.file_utils import read_par_bin
from flow.models.grids import grid_log0

path = "KPZ_1D_twogrids"
p = grid_log0(p_min, p_max, N_p)          # the external grids used by the model
w = grid_log0(w_min, w_max, N_w)
f   = read_par_bin(path, "f.bin", (n_f, N_p, N_w))
par = read_par_bin(path, "flow_parameters.bin", (n_f + 2,))
s, eta_D, eta_nu, g = par.T
G = np.load(path + "/G20_IC_two.npy")
p_two, w_two = flow.p_two, flow.w_two
```

`usage_examples/my_plot_NLO.py` has plotting helpers (e.g. `contourf_Cpw`) used in the notebooks.

