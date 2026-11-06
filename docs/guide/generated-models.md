# Generated LO models

[FRGscalarMSR](https://github.com/GregMRR/FRGscalarMSR) derives LO flow equations symbolically and writes them as a Python module (contract: its `docs/handoff/BMWflow_LO.md`, version 3). `ModelGenerated` integrates any such module; nothing model-specific is coded here:

```python
# In Mathematica: ExportPython[adim, "G" -> alamk^2, "Model" -> "KPZ", "File" -> "eqs/KPZ_LO.py"]
model_params = (2, "LO", dim, grid_ext, grid_int, r, r_)      # n_f must match the module's FUNCTIONS
flow = create.create("eqs/KPZ_LO.py", model_params, "evo1", evo_params)
# or directly: ModelGenerated("eqs/KPZ_LO.py", "LO", dim, grid_ext, grid_int, r, r_,
#                             Integral_extrapol="const", regulators={"rd": (r2, r2_)}, fixed_etas={"etaD": 3.0})
```

- The module gives the functions (`FUNCTIONS`, n_f), their etas, the kernels of the loop integrals, of the eta system and the coupling flow, all in BMWflow's scalar `g`. `ETAS` has one entry per function, `None` for a function without eta (e.g. KPZ's f_λ, whose f(0) = 1 is a Ward identity): the eta vector, the eta system and `IC_NLO.etas_in` then have `model.n_eta` < n_f entries. The adapter applies the measure, fills p = 0 for rows divided by p² (`P2_DIVIDED`, `"const"` or `"quad"`), and solves the eta system.
- Each regulator shape of the module (`SHAPES`) uses `(r, r_)` unless `regulators` names it.
- `fixed_etas` holds the named etas at given values: their rows of the eta system are dropped (so f_X(0) is no longer normalised to 1). Through `create.create`, pass `Integral_extrapol, regulators, fixed_etas` as the 8th–10th entries of `model_params`.
- The adapter refuses a module of another `CONTRACT_VERSION`, of another fixed dimension, or NLO. evo2 is not supported (`Evo2KPZ` is NLO-only).
- Tests: `tests/test_model_generated.py` checks `ModelGenerated` on the vendored `tests/fixtures/KPZ_LO.py` against `ModelKPZ` (LO, d = 1, 2); `tests/test_model_generated_lambda.py` checks `tests/fixtures/KPZLambda_LO.py` (KPZ with a flowing f_λ): at f_λ ≡ 1 it equals the KPZ module, and f_λ(0) stays 1 along a flow (`--lambda-module <file>` checks another export). Run it in a venv: `python -m venv .venv && .venv/bin/pip install -r flow/requirements.txt pytest && .venv/bin/pytest tests` (`--module <file>` checks another export).

