import numpy as np
import scipy.optimize

import linear_invdes
import quadratic_invdes

from typing import Union, Optional

def field_norm_bound_analytical(
        prob_spec: Union[linear_invdes.LinearInvDesProblem,
                    quadratic_invdes.QuadraticInvDesProblem]) -> float:
    """Compute the field norm bound analytically."""
    gfunc = np.linalg.inv(prob_spec.lap_op)
    ref_fld = np.linalg.solve(prob_spec.lap_op, prob_spec.src)
    _, svals, _ = np.linalg.svd(gfunc)
    sval_max = svals[0]
    if sval_max * np.abs(prob_spec.theta_max) >= 1:
        return None
    else:
        return (np.abs(prob_spec.theta_max) * sval_max /
                (1 - np.abs(prob_spec.theta_max) * sval_max))


def field_norm_bound_local(
        prob_spec: Union[linear_invdes.LinearInvDesProblem,
                    quadratic_invdes.QuadraticInvDesProblem],
        theta_init: Optional[np.ndarray] = None) -> float:
    """Compute field norm bounds via local optimization."""
    inc_fld = np.linalg.solve(prob_spec.lap_op, prob_spec.src)
    def _compute_efield(theta) -> np.ndarray:
        maxwell_op = prob_spec.lap_op - np.diag(theta)
        field = np.linalg.solve(maxwell_op, prob_spec.src)
        return field

    def _compute_objective(theta) -> np.ndarray:
        field = _compute_efield(theta)
        return -np.linalg.norm((field - inc_fld))**2

    def _compute_gradient(theta) -> float:
        field = _compute_efield(theta)
        adj_src = np.conj(field - inc_fld)
        adj_field = np.linalg.solve((prob_spec.lap_op - np.diag(theta)).T,
                                    adj_src)
        return -2 * np.real(adj_field * field)


    dim = prob_spec.lap_op.shape[0]
    if theta_init is None:
        theta_init = np.random.uniform(0, prob_spec.theta_max, dim)

    bounds = np.vstack((np.zeros(dim), np.ones(dim) * prob_spec.theta_max)).T
    res = scipy.optimize.minimize(
            _compute_objective, theta_init,
            jac=_compute_gradient, method="L-BFGS-B", bounds=bounds)

    return np.sqrt(-res.fun) / np.linalg.norm(inc_fld)
