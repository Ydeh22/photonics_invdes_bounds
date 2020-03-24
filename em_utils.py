import numpy as np


def maxwell_diff_op_1d(
        dx: float,
        omega: float,
        eps_bg: float,
        dim: int) -> np.ndarray:
    """Setup the one-dimensional maxwell operator.

    Args:
        dx: The pixel size used for discretization.
        omega: The angular frequency of simulation.
        eps_bg: The background permittivity as a scalar.
        dim: The dimensionality of the simulation problem.

    Returns:
        The maxwell operator as a numpy array.
    """
    # Initialize the maxwell operator.
    maxwell_op = np.zeros((dim, dim), dtype=complex)

    # Setup the boundary elements.
    k_bg = omega * np.sqrt(eps_bg)
    maxwell_op[0, 0] = -2 + np.exp(-1.0j * k_bg * dx)
    maxwell_op[dim - 1, dim - 1] = -2 + np.exp(-1.0j * k_bg * dx)

    # Setup the on diagonal and off diagonal elements.
    maxwell_op[np.arange(1, dim - 1), np.arange(1, dim - 1)] = -2
    maxwell_op[np.arange(0, dim - 1), np.arange(1, dim)] = 1
    maxwell_op[np.arange(1, dim), np.arange(0, dim - 1)] = 1

    return -(maxwell_op / (k_bg**2 * dx**2) + np.eye(dim))


def greens_func_op_1d(
        dx: float,
        omega: float,
        eps_bg: float,
        dim: int) -> np.ndarray:
    """Setup the Green's function in 1D.

    Args:
        dx: The pixel size used for discretization.
        omega: The frequency at which the Green's function is being computed.
        eps_bg: The permittivity of the background material.
        dim: The dimensionality of the structure.
    """
    # Calculate the wavenumber corresponding to the background structure.
    k_bg = omega * np.sqrt(eps_bg)

    # Setup the one-dimensional Green's function.
    x_coords = np.arange(dim) * dx
    x_diffs = np.abs(x_coords[:, np.newaxis] - x_coords[np.newaxis, :])
    gfunc = -0.5j * k_bg * dx * np.exp(-1.0j * k_bg * x_diffs)
    return gfunc


def source_diff_1d(
        dx: float,
        omega: float,
        eps_bg: float,
        dim: int) -> np.array:
    """Calculate the source corresponding to the 1D Maxwell problem.

    This assumes that the structure is excited from the left.

    Args:
        dx: The pixel size used for the discretization.
        omega: The frequency of the simulation.
        eps_bg: The background permittivity as a scalar.
        dim: The dimensionality of the problem.

    Returns:
        The source as a numpy array.
    """
    # Initialize the source.
    src = np.zeros(dim, dtype=complex)

    # The source will have only one element corresponding to the incident field
    # on the left of the structure.
    k_bg = omega * np.sqrt(eps_bg)
    src[0] = -np.exp(1.0j * k_bg * dx) + np.exp(-1.0j * k_bg * dx)

    # Normalize the source.
    return -src / (k_bg**2 * dx**2)


def source_pw_1d(
        dx: float,
        omega: float,
        eps_bg: float,
        dim: int) -> np.array:
    """Calculate the source term corresponding to the integral equation."""
    x_coords = np.arange(dim) * dx
    return np.exp(-1.0j * omega * np.sqrt(eps_bg) * x_coords)


def olap_vec_1d(
        dx: float,
        omega: float,
        eps_bg: float,
        dim: int) -> np.array:
    """Calculate the overlap_term."""
    k_bg = omega * np.sqrt(eps_bg)
    x_coords = np.arange(dim) * dx
    return -0.5j * dx * k_bg * np.exp(1.0j * x_coords * k_bg)


def olap_vec_1d_cs(
        dx: float,
        omega: float,
        eps_bg: float,
        dim: int) -> np.array:
    """Calculate the overlap vector corresponding to a cross-section."""
    x_coords = np.arange(dim) * dx
    k_bg = omega * np.sqrt(eps_bg)
    return 0.5j * dx * k_bg * np.exp(-1.0j * x_coords * k_bg)

