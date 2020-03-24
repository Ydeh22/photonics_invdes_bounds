import cvxpy
import numpy as np
import scipy.optimize

from typing import NamedTuple, Optional, Tuple

import cvxpy_complex_utils


class LinearInvDesProblem(NamedTuple):
    """Class to hold the parameters of a linear inverse design problem."""
    dim: int
    lap_op: np.ndarray
    src: np.ndarray
    olap_vec: np.ndarray
    theta_max: float


def locally_solve_primal_problem(
        prob_spec: LinearInvDesProblem,
        theta_init: Optional[np.ndarray] = None
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Locally solve the inverse design problem.

    Note that this function does not cache the simulated electric field. If
    using it for large designs, it is desirable to cache simulated electric
    fields.

    Args:
        prob_spec: The problem specification.

    Returns:
        The locally optimized value of the objective function, the
        corresponding structure and electromagnetic field.
    """
    def _compute_efield(theta) -> np.ndarray:
        """Helper function to compute the electric field."""
        maxwell_op = prob_spec.lap_op - np.diag(theta)
        field = np.linalg.solve(maxwell_op, prob_spec.src)
        return field

    def _compute_objective(theta) -> float:
        """Helper function to compute the value of the quadratic objective."""
        field = _compute_efield(theta)
        return 2 * np.real(np.sum(prob_spec.olap_vec.conj() * theta * field))

    def _compute_gradient(theta) -> float:
        """Helper function to compute the gradient of the objective."""
        # Compute the adjoint field.
        field = _compute_efield(theta)
        adj_src = (prob_spec.olap_vec.conj() * theta)
        adj_field = np.linalg.solve((prob_spec.lap_op - np.diag(theta)).T,
                                    adj_src)

        # Gradient with respect to theta.
        grad_theta = 2 * np.real((prob_spec.olap_vec.conj() * field))
        return grad_theta + 2 * np.real(adj_field * field)

    # Optimize.
    dim = prob_spec.lap_op.shape[0]
    if theta_init is None:
        theta_init =  0 * np.random.uniform(0,
                prob_spec.theta_max, dim)

    bounds = np.vstack((np.zeros(dim), np.ones(dim) * prob_spec.theta_max)).T
    res = scipy.optimize.minimize(
            _compute_objective, theta_init,
            jac=_compute_gradient, method="L-BFGS-B", bounds=bounds)
    return res.fun, res.x, _compute_efield(res.x)



def solve_dual_problem_ptwise(
        prob_spec: LinearInvDesProblem,
        radii: np.ndarray) -> float:
    """Solve the dual problem for a linear objective function."""
    ref_fld = np.linalg.solve(prob_spec.lap_op, prob_spec.src)
    adj_lap_olap = prob_spec.lap_op.conj().T @ prob_spec.olap_vec

    # Setup the dual variables. The complex variables are setup as a tuple of
    # two variables corresponding to the real and imaginary part.
    eta = cvxpy_complex_utils.CvxpyComplexVariable(prob_spec.dim)
    lam = cvxpy.Variable(prob_spec.dim)
    sigma = cvxpy_complex_utils.CvxpyComplexVariable(prob_spec.dim)
    beta = cvxpy.Variable(prob_spec.dim)

    # Setup the objective function.
    obj = -cvxpy.sum(beta) - cvxpy.sum(cvxpy.multiply(radii**2, lam))

    # Setup the constraints.
    constraints = []
    constraints.append(
            eta.real == (np.real(prob_spec.lap_op.T) * sigma.real +
                         np.imag(prob_spec.lap_op.T) * sigma.imag) +
                         np.real(adj_lap_olap))
    constraints.append(
            eta.imag == (np.real(prob_spec.lap_op.T) * sigma.imag -
                         np.imag(prob_spec.lap_op.T) * sigma.real) +
                         np.imag(adj_lap_olap))
    constraints.append(lam >= 0)
    for i in range(prob_spec.dim):
        constraints.append(
                beta[i] >= (cvxpy.quad_over_lin(eta.real[i], lam[i]) +
                            cvxpy.quad_over_lin(eta.imag[i], lam[i])))
        constraints.append(
                beta[i] >= (cvxpy.quad_over_lin(
                        eta.real[i] - prob_spec.theta_max * sigma.real[i],
                        lam[i]) +
                            cvxpy.quad_over_lin(
                                eta.imag[i] - prob_spec.theta_max * sigma.imag[i],
                                lam[i]) +
                        prob_spec.theta_max * (
                            sigma.real[i] * np.real(ref_fld[i]) +
                            sigma.imag[i] * np.imag(ref_fld[i]))))

    prob = cvxpy.Problem(cvxpy.Maximize(obj), constraints)
    prob.solve(cvxpy.ECOS)

    return obj.value


def solve_dual_problem_norm(
        prob_spec: LinearInvDesProblem,
        norm_bound: float) -> float:
    """Solve the dual problem for a linear objective function."""
    ref_fld = np.linalg.solve(prob_spec.lap_op, prob_spec.src)
    adj_lap_olap = prob_spec.lap_op.conj().T @ prob_spec.olap_vec
    radius = norm_bound * np.linalg.norm(ref_fld)

    # Setup the dual variables. The complex variables are setup as a tuple of
    # two variables corresponding to the real and imaginary part.
    eta = cvxpy_complex_utils.CvxpyComplexVariable(prob_spec.dim)
    lam = cvxpy.Variable(1)
    sigma = cvxpy_complex_utils.CvxpyComplexVariable(prob_spec.dim)
    beta = cvxpy.Variable(prob_spec.dim)

    # Setup the objective function.
    obj = -cvxpy.sum(beta) - lam * radius**2

    # Setup the constraints.
    constraints = []
    constraints.append(
            eta.real == (np.real(prob_spec.lap_op.T) * sigma.real +
                         np.imag(prob_spec.lap_op.T) * sigma.imag) +
                         np.real(adj_lap_olap))
    constraints.append(
            eta.imag == (np.real(prob_spec.lap_op.T) * sigma.imag -
                         np.imag(prob_spec.lap_op.T) * sigma.real) +
                         np.imag(adj_lap_olap))
    constraints.append(lam >= 0)
    for i in range(prob_spec.dim):
        constraints.append(
                beta[i] >= (cvxpy.quad_over_lin(eta.real[i], lam) +
                            cvxpy.quad_over_lin(eta.imag[i], lam)))
        constraints.append(
                beta[i] >= (cvxpy.quad_over_lin(
                        eta.real[i] - prob_spec.theta_max * sigma.real[i],
                        lam) +
                            cvxpy.quad_over_lin(
                                eta.imag[i] - prob_spec.theta_max * sigma.imag[i],
                                lam) +
                        prob_spec.theta_max * (
                            sigma.real[i] * np.real(ref_fld[i]) +
                            sigma.imag[i] * np.imag(ref_fld[i]))))

    prob = cvxpy.Problem(cvxpy.Maximize(obj), constraints)
    prob.solve(cvxpy.ECOS)

    return obj.value

