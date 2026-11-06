# Adding a model

For an LO model, prefer generating it ([generated LO models](generated-models.md)). Otherwise:

1. In `flow/models/`, subclass `ModelBase` and implement `Integral_pfixed_dD_LO`, `Integral_pfixed_dD_NLO`, `f_rhs_logder_calc_NLO`, `eta_calc`, `g_rhs_calc`, `calc_upd` and `Integral_upd` (see `model_kpz.py`).
2. For the two grids scheme, in `flow/evolution_two/` subclass `EvolutionTwoBase` and implement `update_IC_two`, `calc_I_inner` and `calc_I_dimful` (see `evo2_kpz.py`).
3. Register both classes in `MODELS` and `EVO_TWO` in `flow/create.py`.

