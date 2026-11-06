<!-- verified: 9cd6db9 2026-11-06 -->
# BMWflow
Python package (`flow/`) that integrates FRG flows of KPZ-type models in the NLO/LO approximation, with the two-grids scheme.
Main author: Liubov Gosteva; GitHub `GregMRR/BMWflowRG`.

## Commands
- Tests: `.venv/bin/pytest tests` (scipy, numba, pytest are in `.venv`, not in /usr/bin/python3); one file:
  `.venv/bin/pytest tests/test_splines.py`; another generated module: `--module <file>` (KPZ with f_λ: `--lambda-module <file>`).
- Docs: `python3 tools/docgen.py reference` (regenerate `docs/reference/`; the pre-commit hook in `.githooks/` runs it),
  `python3 tools/docgen.py check` (CI: stale reference or broken relative link → exit 1). Site: `/project-docs publish`.

## Layout
- `flow/create.py` - factory: `create(model_name, model_params, "evo1"|"evo2", ...)`; registries `MODELS`, `EVO_TWO`.
- `flow/models/` - `ModelBase` (grids, splines, generic RHS) → `ModelKPZ` (NumPy, the one in use), `ModelGenerated` (LO modules
  from FRGscalarMSR). `*_numba*`, `*_jax*`, `model_kpz_jit.py`: unused variants, not imported.
- `flow/evolution.py` - dimensionless Euler integration; `flow/evolution_two/` - second (dimensionful) grid.
- `docs/` - `STATUS.md`, `guide/`, `reference/` (GENERATED from docstrings; `docs/docgen.toml` lists the sources).
- `notes/` - LaTeX derivations of the LO/NLO equations. `usage_examples/` - notebooks.

## Where to change X
- New model: subclass `ModelBase` (+ `EvolutionTwoBase` for evo2), register in `flow/create.py` (`docs/guide/adding-a-model.md`).
- Generated-module format: `flow/models/model_generated.py` reads FRGscalarMSR's `ExportPython` output; the contract
  (`CONTRACT_VERSION`) is FRGscalarMSR's `docs/handoff/BMWflow_LO.md`. A format change updates both repos and re-vendors
  `tests/fixtures/KPZ_LO.py` and `tests/fixtures/KPZLambda_LO.py`.
- A public symbol's docs: its docstring (NumPy style); features and limits: `docs/STATUS.md`.

## Gotchas
- `.gitignore` ignores every dot-path (`.*`); `.githooks/` and `.github/` are re-included explicitly.
- `Evolution` prompts on stdin when the output directory exists: scripts must use a new directory.
- Regulators take q, not q²; their `name_` derivatives are with respect to q² (except the test regulator `exp`).
