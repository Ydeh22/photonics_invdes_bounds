import cvxpy
import numpy as np
import scipy.optimize

from typing import NamedTuple, Optional, Tuple

import cvxpy_complex_utils


class QuadraticInvDesProblem(NamedTuple):
    """Class to hold the parameters for a quadratic inverse design problem.

    Parameters:
        lap_op: The laplacian operator specified as a numpy array.
            TODO(@rtrivedi) - change this to a sparse specification.
        src: The source corresponding to the given excitation.
        olap_vec: The overlap vector to be applied on the susceptibility vector
            to calculate the bound.
        target_amp: The desired target for the optimization problem.
        theta_max: The susceptibility that is allowed relative to the background
            permittivity.
    """
    dim: int
    lap_op: np.ndarray
    src: np.ndarray
    olap_vec: np.ndarray
    target_amp: np.ndarray
    theta_max: float


def locally_solve_primal_problem(
        prob_spec: QuadraticInvDesProblem,
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
        return np.abs(np.sum(prob_spec.olap_vec.conj() * theta * field) -
                      prob_spec.target_amp)**2

    def _compute_gradient(theta) -> float:
        """Helper function to compute the gradient of the objective."""
        # Compute the adjoint field.
        field = _compute_efield(theta)
        comp_olap = (np.sum(prob_spec.olap_vec.conj() * theta * field) -
                     prob_spec.target_amp)
        adj_src = np.conj(comp_olap) * (prob_spec.olap_vec.conj() * theta)
        adj_field = np.linalg.solve((prob_spec.lap_op - np.diag(theta)).T,
                                    prob_spec.olap_vec.conj() * theta)

        # Gradient with respect to theta.
        grad_theta = 2 * np.real(
                np.conj(comp_olap) * (prob_spec.olap_vec.conj() * field))
        return grad_theta + 2 * np.real(adj_field * field)

    # Optimize.
    dim = prob_spec.lap_op.shape[0]
    if theta_init is None:
        theta_init = np.random.uniform(0, prob_spec.theta_max, dim)

    bounds = np.vstack((np.zeros(dim), np.ones(dim) * prob_spec.theta_max)).T
    res = scipy.optimize.minimize(
            _compute_objective, theta_init,
            jac=_compute_gradient, method="L-BFGS-B", bounds=bounds)
    return res.fun, res.x, _compute_efield(res.x)


def solve_dual_problem(
        prob_spec: QuadraticInvDesProblem,
        norm_bound: float,
        ref_fld: Optional[np.ndarray] = None) -> float:
    """Solve the dual problem for a quadratic objective.

    For a valid solution, the dual problem requires specification of a reference
    field and a radii within the reference field that the simulated field can
    be allowed to lie within.

    Args:
        prob_spec: The problem specification for the primal problem.
        ref_fld: The reference field specified as a numpy array.
        radii: The radius within which the simulated field can lie within the
            reference field.

    Returns:
        The result of the dual problem.
    """
    # If the reference field is not specified, use the solution of Maxwell's
    # equations in the background material.
    if ref_fld is None:
        ref_fld = np.linalg.solve(prob_spec.lap_op, prob_spec.src)
        ref_src = prob_spec.src
    else:
        ref_src = prob_spec.lap_op @ ref_fld
    radius = np.linalg.norm(ref_fld) * norm_bound

    # Setup the dual variables. The complex variables are setup as a tuple of
    # two variables corresponding to the real and imaginary part.
    nu = cvxpy_complex_utils.CvxpyComplexVariable(prob_spec.dim)
    eta = cvxpy_complex_utils.CvxpyComplexVariable(prob_spec.dim)
    # lam = cvxpy.Variable(prob_spec.dim)
    lam = cvxpy.Variable(1)
    sigma = cvxpy_complex_utils.CvxpyComplexVariable(prob_spec.dim)
    beta = cvxpy.Variable(prob_spec.dim)
    alpha = cvxpy_complex_utils.CvxpyComplexVariable(1)

    # Setup the objective function.
    obj = (cvxpy.sum(cvxpy.multiply(np.real(ref_src - prob_spec.src), nu.real)) +
           cvxpy.sum(cvxpy.multiply(np.imag(ref_src - prob_spec.src), nu.imag)) +
           np.abs(prob_spec.target_amp)**2 -
           cvxpy.power(alpha.real - np.real(prob_spec.target_amp), 2) -
           cvxpy.sum(beta) - cvxpy.multiply(radius**2, lam))

    # Setup the constraints.
    constraints = []
    constraints.append(
            eta.real == (np.real(prob_spec.lap_op.T) * nu.real +
                         np.imag(prob_spec.lap_op.T) * nu.imag))
    constraints.append(
            eta.imag == (np.real(prob_spec.lap_op.T) * nu.imag -
                         np.imag(prob_spec.lap_op.T) * nu.real))
    constraints.append(
            (sigma.real - nu.real) == (alpha.real * prob_spec.olap_vec.real -
                                       alpha.imag * prob_spec.olap_vec.imag))
    constraints.append(
            (sigma.imag - nu.imag) == (alpha.real * prob_spec.olap_vec.imag +
                                       alpha.imag * prob_spec.olap_vec.real))
    constraints.append(lam >= 0)
    for i in range(prob_spec.dim):
        constraints.append(
                beta[i] >= (cvxpy.quad_over_lin(nu.real[i], lam) +
                            cvxpy.quad_over_lin(nu.imag[i], lam)))
        constraints.append(
                beta[i] >= (cvxpy.quad_over_lin(
                        nu.real[i] - prob_spec.theta_max * sigma.real[i],
                        lam) +
                            cvxpy.quad_over_lin(
                                nu.imag[i] - prob_spec.theta_max * sigma.imag[i],
                                lam) +
                        prob_spec.theta_max * (
                            sigma.real[i] * np.real(ref_fld[i]) +
                            sigma.imag[i] * np.imag(ref_fld[i]))))

    # Final problem and its solution.
    prob = cvxpy.Problem(cvxpy.Maximize(obj), constraints)
    prob.solve(cvxpy.ECOS)

    return obj.value
