import numpy as np


def prism_f(x, y, z, r=None):
    """
    Evaluates the indefinite volume integral for the 1/r kernel.

    This is used to evaluate the gravitational potential of dense prisms.

    Parameters
    ----------
    x, y, z : (...) numpy.ndarray
        The nodal locations to evaluate the function at
    r : (...) numpy.ndarray, optional
        The distance of each node location to the origin (if pre-computed)

    Returns
    -------
    (...) numpy.ndarray
    """
    if r is None:
        r = np.sqrt(x * x + y * y + z * z)
    out = np.zeros_like(r)

    nz_x = x != 0.0
    nz_y = y != 0.0
    nz_z = z != 0.0

    nz = nz_x & nz_y
    out[nz] -= x[nz] * y[nz] * np.log(z[nz] + r[nz])

    nz = nz_y & nz_z
    out[nz] -= y[nz] * z[nz] * np.log(x[nz] + r[nz])

    nz = nz_x & nz_z
    out[nz] -= x[nz] * z[nz] * np.log(y[nz] + r[nz])

    out[nz_x] += 0.5 * x[nz_x] * x[nz_x] * np.arctan(y[nz_x] * z[nz_x] / (x[nz_x] * r[nz_x]))
    out[nz_y] += 0.5 * y[nz_y] * y[nz_y] * np.arctan(x[nz_y] * z[nz_y] / (y[nz_y] * r[nz_y]))
    out[nz_z] += 0.5 * z[nz_z] * z[nz_z] * np.arctan(x[nz_z] * y[nz_z] / (z[nz_z] * r[nz_z]))
    return out


def prism_fz(x, y, z, r=None):
    """
    Evaluates the indefinite volume integral for the d/dz * 1/r kernel.

    This is used to evaluate the gravitational field of dense prisms.

    Parameters
    ----------
    x, y, z : (...) numpy.ndarray
        The nodal locations to evaluate the function at
    r : (...) numpy.ndarray, optional
        The distance of each node location to the origin (if pre-computed)

    Returns
    -------
    (...) numpy.ndarray

    Notes
    -----
    Can be used to compute other directions by cycling the inputs
    """
    if r is None:
        r = np.sqrt(x * x + y * y + z * z)
    out = np.zeros_like(r)

    nz = x != 0.0
    out[nz] += x[nz] * np.log(y[nz] + r[nz])

    nz = y != 0.0
    out[nz] += y[nz] * np.log(x[nz] + r[nz])

    nz = z != 0.0
    out[nz] -= z[nz] * np.arctan(x[nz]*y[nz]/(z[nz]*r[nz]))

    return out


def prism_fzz(x, y, z, r=None):
    """
    Evaluates the indefinite volume integral for the d**2/dz**2 * 1/r kernel.

    This is used to evaluate the gravitational field of dense prisms.

    Parameters
    ----------
    x, y, z : (...) numpy.ndarray
        The nodal locations to evaluate the function at
    r : (...) numpy.ndarray, optional
        The distance of each node location to the origin (if pre-computed)

    Returns
    -------
    (...) numpy.ndarray

    Notes
    -----
    Can be used to compute other directions by cycling the inputs.
    """
    if r is None:
        r = np.sqrt(x * x + y * y + z * z)

    out = np.zeros_like(r)
    nz = z != 0.0
    out[nz] = np.arctan(x[nz] * y[nz] / (z[nz] * r[nz]))
    return out


def prism_fzx(x, y, z, r=None):
    """
    Evaluates the indefinite volume integral for the d**2/(dz*dx) * 1/r kernel.

    This is used to evaluate the gravitational field of dense prisms.

    Parameters
    ----------
    x, y, z : (...) numpy.ndarray
        The nodal locations to evaluate the function at
    r : (...) numpy.ndarray, optional
        The distance of each node location to the origin (if pre-computed)

    Returns
    -------
    (...) numpy.ndarray

    Notes
    -----
    Can be used to compute other directions by cycling the inputs.
    """
    if r is None:
        r = np.sqrt(x * x + y * y + z * z)

    out = np.zeros_like(r)
    v = y + r

    nz = (v == 0) & (y < 0)
    out[nz] = np.log(-2 * y[nz])

    nz = v != 0
    out[nz] = -np.log(v[nz])
    return out


def prism_fzy(x, y, z, r=None):
    """
    Evaluates the indefinite volume integral for the d**2/(dz*dx) * 1/r kernel.

    This is used to evaluate the gravitational field of dense prisms.

    Parameters
    ----------
    x, y, z : (...) numpy.ndarray
        The nodal locations to evaluate the function at
    r : (...) numpy.ndarray, optional
        The distance of each node location to the origin (if pre-computed)

    Returns
    -------
    (...) numpy.ndarray

    Notes
    -----
    Can be used to compute other directions by cycling the inputs.
    """
    if r is None:
        r = np.sqrt(x * x + y * y + z * z)

    out = np.zeros_like(r)
    v = x + r

    nz = (v == 0) & (x < 0)
    out[nz] = np.log(-2 * x[nz])

    nz = v != 0
    out[nz] = -np.log(v[nz])
    return out


def prism_fzzz(x, y, z, r=None):
    """
    Evaluates the indefinite volume integral for the d**3/(dz**3) * 1/r kernel.

    This is used to evaluate the gravitational field of dense prisms.

    Parameters
    ----------
    x, y, z : (...) numpy.ndarray
        The nodal locations to evaluate the function at
    r : (...) numpy.ndarray, optional
        The distance of each node location to the origin (if pre-computed)

    Returns
    -------
    (...) numpy.ndarray

    Notes
    -----
    Can be used to compute other directions by cycling the inputs.
    """
    if r is None:
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


def prism_fxxy(x, y, z, r=None):
    """
    Evaluates the indefinite volume integral for the d**3/(dx**2 * dy) * 1/r kernel.

    This is used to evaluate the gravitational field of dense prisms.

    Parameters
    ----------
    x, y, z : (...) numpy.ndarray
        The nodal locations to evaluate the function at
    r : (...) numpy.ndarray, optional
        The distance of each node location to the origin (if pre-computed)

    Returns
    -------
    (...) numpy.ndarray

    Notes
    -----
    Can be used to compute other directions by cycling the inputs.
    """
    if r is None:
        r = np.sqrt(x * x + y * y + z * z)

    v = x * x + y * y
    out = np.zeros_like(r)
    nz = v != 0.0
    out[nz] = 1 / v[nz]

    nz = r != 0.0
    out[nz] *= -x[nz] * z[nz] / r[nz]
    return out


def prism_fxxz(x, y, z, r=None):
    """
    Evaluates the indefinite volume integral for the d**3/(dx**2 * dz) * 1/r kernel.

    This is used to evaluate the gravitational field of dense prisms.

    Parameters
    ----------
    x, y, z : (...) numpy.ndarray
        The nodal locations to evaluate the function at
    r : (...) numpy.ndarray, optional
        The distance of each node location to the origin (if pre-computed)

    Returns
    -------
    (...) numpy.ndarray

    Notes
    -----
    Can be used to compute other directions by cycling the inputs.
    """
    if r is None:
        r = np.sqrt(x * x + y * y + z * z)

    v = x * x + z * z
    out = np.zeros_like(r)
    nz = v != 0.0
    out[nz] = 1 / v[nz]

    nz = r != 0.0
    out[nz] *= -x[nz] * y[nz] / r[nz]
    return out


def prism_fxyz(x, y, z, r=None):
    """
    Evaluates the indefinite volume integral for the d**3/(dx * dy * dz) * 1/r kernel.

    This is used to evaluate the gravitational field of dense prisms.

    Parameters
    ----------
    x, y, z : (...) numpy.ndarray
        The nodal locations to evaluate the function at
    r : (...) numpy.ndarray, optional
        The distance of each node location to the origin (if pre-computed)

    Returns
    -------
    (...) numpy.ndarray

    Notes
    -----
    Can be used to compute other directions by cycling the inputs.
    """
    if r is None:
        r = np.sqrt(x * x + y * y + z * z)
    out = np.zeros_like(r)
    nz = r != 0.0
    out[nz] = 1 / r[nz]

    return out
