import numpy as np
from scipy.special import xlogy

__all__ = [
    "prism_f",
    "prism_fz",
    "prism_fzz",
    "prism_fzx",
    "prism_fzy",
    "prism_fzzz",
    "prism_fxxy",
    "prism_fxxz",
    "prism_fxyz",
]


def _plus_r(a, b, c, r):
    """a + r, with r = sqrt(a**2 + b**2 + c**2), without cancellation for a < 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a < 0, (b * b + c * c) / (r - a), a + r)


def _prism_f(x, y, z):
    """
    Evaluates the indefinite volume integral for the 1/r kernel.

    This is used to evaluate the gravitational potential of dense prisms.

    Parameters
    ----------
    x, y, z : (...) numpy.ndarray
        The nodal locations to evaluate the function at

    Returns
    -------
    (...) numpy.ndarray
    """
    r = np.sqrt(x * x + y * y + z * z)
    out = np.zeros_like(r)

    nz_x = x != 0.0
    nz_y = y != 0.0
    nz_z = z != 0.0

    temp = _plus_r(z, x, y, r)
    nz = nz_x & nz_y & (temp > 0.0)
    out[nz] -= x[nz] * y[nz] * np.log(temp[nz])

    temp = _plus_r(x, y, z, r)
    nz = nz_y & nz_z & (temp > 0.0)
    out[nz] -= y[nz] * z[nz] * np.log(temp[nz])

    temp = _plus_r(y, x, z, r)
    nz = nz_x & nz_z & (temp > 0.0)
    out[nz] -= x[nz] * z[nz] * np.log(temp[nz])

    out[nz_x] += 0.5 * x[nz_x] * x[nz_x] * np.arctan(y[nz_x] * z[nz_x] / (x[nz_x] * r[nz_x]))
    out[nz_y] += 0.5 * y[nz_y] * y[nz_y] * np.arctan(x[nz_y] * z[nz_y] / (y[nz_y] * r[nz_y]))
    out[nz_z] += 0.5 * z[nz_z] * z[nz_z] * np.arctan(x[nz_z] * y[nz_z] / (z[nz_z] * r[nz_z]))
    return out


def _prism_fz(x, y, z):
    """
    Evaluates the indefinite volume integral for the d/dz * 1/r kernel.

    This is used to evaluate the gravitational field of dense prisms.

    Parameters
    ----------
    x, y, z : (...) numpy.ndarray
        The nodal locations to evaluate the function at

    Returns
    -------
    (...) numpy.ndarray

    Notes
    -----
    Can be used to compute other components by cycling the inputs
    """
    r = np.sqrt(x * x + y * y + z * z)
    out = np.zeros_like(r)

    temp = _plus_r(y, x, z, r)
    nz = (x != 0.0) & (temp > 0)
    out[nz] += x[nz] * np.log(temp[nz])

    temp = _plus_r(x, y, z, r)
    nz = (y != 0.0) & (temp > 0)
    out[nz] += y[nz] * np.log(temp[nz])

    nz = z != 0.0
    out[nz] -= z[nz] * np.arctan(x[nz]*y[nz]/(z[nz]*r[nz]))

    return out


def _prism_fzz(x, y, z):
    """
    Evaluates the indefinite volume integral for the d**2/dz**2 * 1/r kernel.

    This is used to evaluate the gravitational field of dense prisms.

    Parameters
    ----------
    x, y, z : (...) numpy.ndarray
        The nodal locations to evaluate the function at

    Returns
    -------
    (...) numpy.ndarray

    Notes
    -----
    Can be used to compute other components by cycling the inputs.
    """
    r = np.sqrt(x * x + y * y + z * z)

    out = np.zeros_like(r)
    nz = z != 0.0
    out[nz] = np.arctan(x[nz] * y[nz] / (z[nz] * r[nz]))
    return out


def _prism_fzx(x, y, z):
    """
    Evaluates the indefinite volume integral for the d**2/(dz*dx) * 1/r kernel.

    This is used to evaluate the gravitational gradient of dense prisms.

    Parameters
    ----------
    x, y, z : (...) numpy.ndarray
        The nodal locations to evaluate the function at

    Returns
    -------
    (...) numpy.ndarray

    Notes
    -----
    Can be used to compute other components by cycling the inputs.
    """
    r = np.sqrt(x * x + y * y + z * z)

    out = np.zeros_like(r)
    # y + r cancels near the negative y axis; there it equals
    # (x**2 + z**2) / (r - y). On the axis the kernel diverges as -2 ln(d)
    # with the distance d to it; keep the finite part.
    neg = y < 0
    temp = x * x + z * z
    on = neg & (temp == 0)
    out[on] = np.log(-2 * y[on])
    near = neg & (temp != 0)
    out[near] = np.log((r[near] - y[near]) / temp[near])

    v = y + r
    pos = ~neg & (v != 0)
    out[pos] = -np.log(v[pos])
    return out


def _prism_fzy(x, y, z):
    """
    Evaluates the indefinite volume integral for the d**2/(dz*dx) * 1/r kernel.

    This is used to evaluate the gravitational gradient of dense prisms.

    Parameters
    ----------
    x, y, z : (...) numpy.ndarray
        The nodal locations to evaluate the function at

    Returns
    -------
    (...) numpy.ndarray

    Notes
    -----
    Can be used to compute other components by cycling the inputs.
    """
    r = np.sqrt(x * x + y * y + z * z)

    out = np.zeros_like(r)
    # x + r cancels near the negative x axis; there it equals
    # (y**2 + z**2) / (r - x). On the axis the kernel diverges as -2 ln(d)
    # with the distance d to it; keep the finite part.
    neg = x < 0
    temp = y * y + z * z
    on = neg & (temp == 0)
    out[on] = np.log(-2 * x[on])
    near = neg & (temp != 0)
    out[near] = np.log((r[near] - x[near]) / temp[near])

    v = x + r
    pos = ~neg & (v != 0)
    out[pos] = -np.log(v[pos])
    return out


def _prism_fzzz(x, y, z):
    """
    Evaluates the indefinite volume integral for the d**3/(dz**3) * 1/r kernel.

    This is used to evaluate the magnetic gradient of susceptible prisms.

    Parameters
    ----------
    x, y, z : (...) numpy.ndarray
        The nodal locations to evaluate the function at

    Returns
    -------
    (...) numpy.ndarray

    Notes
    -----
    Can be used to compute other components by cycling the inputs.
    """
    r = np.sqrt(x * x + y * y + z * z)

    v2 = x * x + z * z
    v3 = y * y + z * z
    out = np.zeros_like(r)
    nz = v2 != 0.0
    out[nz] = 1 / v2[nz]
    nz = v3 != 0.0
    out[nz] += 1 / v3[nz]

    nz = r != 0.0
    out[nz] *= x[nz] * y[nz] / r[nz]
    return out


def _prism_fxxy(x, y, z):
    """
    Evaluates the indefinite volume integral for the d**3/(dx**2 * dy) * 1/r kernel.

    This is used to evaluate the magnetic gradient of susceptible prisms.

    Parameters
    ----------
    x, y, z : (...) numpy.ndarray
        The nodal locations to evaluate the function at

    Returns
    -------
    (...) numpy.ndarray

    Notes
    -----
    Can be used to compute other components by cycling the inputs.
    """
    r = np.sqrt(x * x + y * y + z * z)

    v = x * x + y * y
    out = np.zeros_like(r)
    nz = v != 0.0
    out[nz] = 1 / v[nz]

    nz = r != 0.0
    out[nz] *= -x[nz] * z[nz] / r[nz]
    return out


def _prism_fxxz(x, y, z):
    """
    Evaluates the indefinite volume integral for the d**3/(dx**2 * dz) * 1/r kernel

    This is used to evaluate the magnetic gradient of susceptible prisms.

    Parameters
    ----------
    x, y, z : (...) numpy.ndarray
        The nodal locations to evaluate the function at

    Returns
    -------
    (...) numpy.ndarray

    Notes
    -----
    Can be used to compute other components by cycling the inputs.
    """
    r = np.sqrt(x * x + y * y + z * z)

    v = x * x + z * z
    out = np.zeros_like(r)
    nz = v != 0.0
    out[nz] = 1 / v[nz]

    nz = r != 0.0
    out[nz] *= -x[nz] * y[nz] / r[nz]
    return out


def _prism_fxyz(x, y, z):
    """
    Evaluates the indefinite volume integral for the d**3/(dx * dy * dz) * 1/r kernel.

    This is used to evaluate the magnetic gradient of susceptible prisms.

    Parameters
    ----------
    x, y, z : (...) numpy.ndarray
        The nodal locations to evaluate the function at

    Returns
    -------
    (...) numpy.ndarray

    Notes
    -----
    Can be used to compute other components by cycling the inputs.
    """
    r = np.sqrt(x * x + y * y + z * z)
    out = np.zeros_like(r)
    nz = r != 0.0
    out[nz] = 1 / r[nz]

    return out


try:
    from geoana.kernels._extensions.potential_field_prism import (
        prism_f,
        prism_fz,
        prism_fzz,
        prism_fzx,
        prism_fzy,
        prism_fzzz,
        prism_fxxy,
        prism_fxxz,
        prism_fxyz,
    )
except ImportError:
    # Store the above as the kernels
    prism_f = _prism_f
    prism_fz = _prism_fz
    prism_fzz = _prism_fzz
    prism_fzx = _prism_fzx
    prism_fzy = _prism_fzy
    prism_fzzz = _prism_fzzz
    prism_fxxy = _prism_fxxy
    prism_fxxz = _prism_fxxz
    prism_fxyz = _prism_fxyz
