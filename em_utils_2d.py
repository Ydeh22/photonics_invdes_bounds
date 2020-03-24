import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from typing import Tuple


def _dirchlet_laplacian_2d(
        dx: float,
        omega: float,
        eps_bg: float,
        dims: Tuple[int, int]) -> scipy.sparse.linalg.LinearOperator:
    """Sets up the laplacian assuming dirchlet boundary conditions."""
    k_bg = omega * np.sqrt(eps_bg)
    def _matvecprod(field: np.array) -> np.array:
        field_grid = np.reshape(field, newshape=dims)
        field_out = np.zeros_like(field_grid)
        field_out[1:-1, 1:-1] = (field[0:-2, 1:-1] +
                                 field[2:, 1:-1] +
                                 field[1:-1, 0:-2] +
                                 field[1:-1, 2:] -
                                 4 * field[1:-1, 1:-1]) / (dx**2)
        return field_out

    return scipy.sparse.LinearOperator(shape=(np.prod(dims), np.prod(dims)),
                                       matvec=_matvecprod)

def _compute_centroidal_appx(
        row_ind: Tuple[int, int],
        col_ind: Tuple[int, int],
        surf_normal: Tuple[float, float],
        k_bg: float) -> complex:
    """Compute the matrix element corresponding to different segments."""
    pass



def _boundary_efie_2d(
        dx: float,
        omega: float,
        eps_bg: float,
        dims: float) -> scipy.sparse.linalg.LinearOperator:
    """Sets up the boundary integral equation in 2d."""
    k_bg = omega * np.sqrt(eps_bg)
    def _matvecprod(field: np.array) -> np.array:
        field_grid = np.reshape(field, newshape=dims)
    pass

def boundary_integral_laplacian_2d(
    dx: float,
    omega: float,
    eps_bg: float,
    dims: Tuple[int, int]) -> scipy.sparse.linalg.LinearOperator:
    """Sets up the boundary integral laplacian for 2d systems."""
    # This needs to be done in two parts, one is imposing the differential
    # equation at the center and the other is the boundary integral equation.
    def _matvecprod_dirchlet(field: np.array) -> np.array:
        """Calculate the matrix-vector product for assuming direchlet BC."""
        field_grid = np.reshape(field, newshape=dims)
    pass


def green_func_2d(
        dx: float,
        omega: float,
        eps_bg: float,
        dims: Tuple[int, int],
        num_iterp_pts: int = 500) -> np.ndarray:
    """Setup the Green's function for the 2D problem."""
    k_bg = omega * np.sqrt(eps_bg)
    x_coords = np.arange(dims[0]) * dx
    y_coords = np.arange(dims[1]) * dx
    # Calculate the Hankel function at all required values.
    min_arg = 0.5 * dx * k_bg
    max_arg = 1.5 * np.sqrt((dims[0] - 1)**2 + (dims[1] - 1)**2) * dx * k_bg
    args = np.linspace(min_arg, max_arg, num_iterp_pts)
    hf = scipy.special.hankel2(0, args)
    hf_interp = scipy.interpolate.interp1d(args, hf, bounds_error=False, fill_value=0)

    # Calculate the Hankel function.
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    R = np.sqrt((x_grid.flatten()[:, np.newaxis] -
                 x_grid.flatten()[np.newaxis, :])**2 +
                (y_grid.flatten()[:, np.newaxis] -
                 y_grid.flatten()[np.newaxis, :])**2)
    hf_on_grid = hf_interp(k_bg * R)

    # Calculate the off-diagonal element.
    dr = dx / np.sqrt(np.pi)
    gfunc = -0.5j * np.pi * k_bg * dr * (scipy.special.jv(1, k_bg * dr) * 
                                         hf_on_grid)
    gfunc_diag = (-0.5j * np.pi * k_bg * dr * (scipy.special.hankel2(1, k_bg * dr))
                   - 1.0)
    gfunc += np.diag(np.ones(np.prod(dims)) * gfunc_diag)
    return gfunc

def inc_field_2d(
        dx: float,
        angle: float,
        omega: float,
        eps_bg: float,
        dims: Tuple[int, int]) -> np.array:
    k_bg = omega * np.sqrt(eps_bg)
    x_coords = np.arange(dims[0]) * dx
    y_coords = np.arange(dims[1]) * dx
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    return np.exp(-1.0j * k_bg * (x_grid * np.cos(angle) + y_grid * np.sin(angle))).flatten()


def plane_wave_olap_2d(
        dx: float,
        angle: float,
        omega: float,
        eps_bg: float,
        dims: Tuple[int, int]) -> np.array:
    k_bg = omega * np.sqrt(eps_bg)
    x_coords = np.arange(dims[0]) * dx
    y_coords = np.arange(dims[1]) * dx
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    return (0.5j * dx**2 * k_bg * np.exp(-1.0j * k_bg * (x_grid * np.cos(angle) + y_grid * np.sin(angle)))).flatten()


def green_func_tm_2d(
        dx: float,
        omega: float,
        eps_bg: float,
        dims: Tuple[int, int],
        num_interp_pts: int = 500) -> np.ndarray:
    """Calculating the Green's function in TM."""
    # Setup the coordinates.
    x_coords = np.arange(dims[0]) * dx
    y_coords = np.arange(dims[1]) * dx
    # Wavenumber.
    k_bg = omega * np.sqrt(eps_bg)

    # Setup the bessel and hankel functions for interpolation.
    min_args = 0.5 * dx * k_bg
    max_args = 1.5 * np.sqrt((dims[0] - 1)**2 +
                             (dims[1] - 1)**2) * dx * k_bg
    args = np.linspace(min_args, max_args, num_interp_pts)
    # Compute the hankel functions of the zeroth, first and second order.
    hf_0 = scipy.special.hankel2(0, args)
    hf_1 = scipy.special.hankel2(1, args)
    # Interpolate the hankel functions and their derivatives.
    hf_interp_0 = scipy.interpolate.interp1d(args, hf_0, bounds_error=False,
                                             fill_value=0)
    hf_interp_1 = scipy.interpolate.interp1d(args, hf_1, bounds_error=False,
                                             fill_value=0)

    # Compute the matrix of distances between different points on grid.
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()
    x_diff = x_grid[:, np.newaxis] - x_grid[np.newaxis, :]
    y_diff = y_grid[:, np.newaxis] - y_grid[np.newaxis, :]
    R = np.sqrt(x_diff**2 + y_diff**2)
    # For convenience, we replace the on diagonal elements of R, which
    # correspond to distance between the same points with a very small number.
    np.fill_diagonal(R, 1e-6)
    # Setup the matrices the matrices corresponding to the derivatives of the
    # Green's function integral over the circular cells.
    dr = dx / np.sqrt(np.pi)
    # Calculate the integral.
    K = -0.5j * np.pi * dr * scipy.special.jv(1, k_bg * dr) / R**3
    G_xx = K * (k_bg * R * y_diff**2 * hf_interp_0(k_bg * R) +
                (x_diff**2 - y_diff**2) * hf_interp_1(k_bg * R))
    G_xy = K * x_diff * y_diff * (2 * hf_interp_1(k_bg * R) -
                                  k_bg * R * hf_interp_0(k_bg * R))
    G_yy = K * (k_bg * R * x_diff**2 * hf_interp_0(k_bg * R) +
                (y_diff**2 - x_diff**2) * hf_interp_1(k_bg * R))
    np.fill_diagonal(G_xy, 0)
    np.fill_diagonal(G_xx, -0.25j * np.pi * k_bg * dr * scipy.special.hankel2(1, k_bg * dr) - 1)
    np.fill_diagonal(G_yy, -0.25j * np.pi * k_bg * dr * scipy.special.hankel2(1, k_bg * dr) - 1)
    # Assemble the full coefficient matrix.
    G = np.block([[G_xx, G_xy],
                  [G_xy, G_yy]])
    return G


def plane_wave_2d_tm(
        dx: float,
        angle: float,
        omega: float,
        eps_bg: float,
        dims: Tuple[int, int]) -> np.ndarray:
    x_coords = np.arange(dims[0]) * dx
    y_coords = np.arange(dims[1]) * dx
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()
    phase = omega * np.sqrt(eps_bg) * (x_grid * np.cos(angle) +
                                       y_grid * np.sin(angle))
    Ex = np.sin(angle) * np.exp(-1.0j * phase)
    Ey = np.cos(angle) * np.exp(-1.0j * phase)
    return np.hstack([Ex, Ey])

