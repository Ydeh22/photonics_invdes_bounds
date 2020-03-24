"""Module for some simple utilities to handle complex variables with cvxpy."""
import cvxpy


class CvxpyComplexVariable:
    def __init__(self, dim: int) -> None:
        self._real = cvxpy.Variable(dim)
        self._imag = cvxpy.Variable(dim)

    @property
    def real(self) -> cvxpy.Variable:
        return self._real

    @property
    def imag(self) -> cvxpy.Variable:
        return self._imag


