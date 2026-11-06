"""Parameter containers: external and internal grids, initial condition of the flow."""
from dataclasses import dataclass
import numpy as np

from .flow_types import REAL


@dataclass
class ParamsGridExternal:
    """External grids of momentum p and frequency w, on which the flowing functions are stored.

    Attributes
    ----------
    N_p : int, default 100
        Points in the p grid, including the prepended p = 0.
    p_max : REAL, default 1e2
        Largest p.
    p_min : REAL, default 1e-2
        Smallest nonzero p.
    N_w : int, default 1
        Points in the w grid, including w = 0 (NLO only; LO uses w = [0]).
    w_max : REAL, default 0
        Largest w (NLO only; set it for NLO).
    w_min : REAL, default 0
        Smallest nonzero w (NLO only; set it for NLO).
    scale : str, default 'log'
        Only 'log' is supported: [0, geomspace(min, max, N - 1)]."""
    N_p: int = 100
    p_max: REAL = 1E2
    p_min: REAL = 1E-2
    N_w: int = 1
    w_max: REAL = 0
    w_min: REAL = 0
    scale: str = 'log'


@dataclass
class ParamsGridInternal:
    """Integration grids of the loop momentum q and the angle theta.

    The field order is deg_q, q_max, deg_theta, theta_max: keyword arguments are safer.

    Attributes
    ----------
    deg_q : int, default 40
        Gauss-Legendre nodes on q in [0, q_max].
    q_max : REAL, default 10
        Cut-off of the loop momentum: pick it where the regulator is negligible (4 for the Wetterich-type regulators).
    deg_theta : int, default 10
        Gauss-Legendre nodes on theta in [0, theta_max]; ignored for dim = 1.
    theta_max : REAL, default pi
        Upper angle; ignored for dim = 1."""
    deg_q: int = 40
    q_max: REAL = 10
    deg_theta: int = 10
    theta_max: REAL = np.pi


@dataclass
class IC_NLO:
    """Initial condition of the flow at kappa = Lambda (used in LO and NLO).

    Attributes
    ----------
    g_in : REAL
        Initial dimensionless coupling.
    etas_in : np.ndarray
        Initial anomalous dimensions, shape (n_f,). For KPZ: zeros.
    fs_in : np.ndarray
        Initial dimensionless functions, shape (n_f, N_p, N_w), with N_w = 1 in LO. For KPZ: ones."""
    g_in: REAL
    etas_in: np.ndarray
    fs_in: np.ndarray
