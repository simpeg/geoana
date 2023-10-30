cimport cython

from libc.math cimport sqrt, log, atan

@cython.cdivision
@cython.ufunc
cdef api double prism_f(double x, double y, double z) nogil:
    """Evaluates the indefinite volume integral for the 1/r kernel.

    This is used to evaluate the gravitational potential of dense prisms.

    Parameters
    ----------
    x, y, z : (...) numpy.ndarray
        The nodal locations to evaluate the function at

    Returns
    -------
    (...) numpy.ndarray
    """
    cdef:
        double v = 0.0
        double r, temp

    r = sqrt(x * x + y * y + z * z)
    if x != 0.0 and y != 0.0:
        temp = z + r
        if temp > 0.0:
            v -= x * y * log(temp)
        v += 0.5 * x * x * atan( y * z / (x * r))
    if y != 0.0 and z != 0.0:
        temp = x + r
        if temp > 0.0:
            v -= y * z * log(temp)
        v += 0.5 * y * y * atan(z * x / (y * r))
    if z != 0.0 and x != 0.0:
        temp = y + r
        if temp > 0.0:
            v -= z * x * log(temp)
        v += 0.5 * z * z * atan(x * y / (z * r))
    return v

@cython.cdivision
@cython.ufunc
cdef api double prism_fz(double x, double y, double z) nogil:
    """Evaluates the indefinite volume integral for the d/dz * 1/r kernel.

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
    cdef:
        double v = 0.0
        double r, temp

    r = sqrt(x * x + y * y + z * z)
    if x != 0.0:
        temp = y + r
        if temp > 0.0:
            v += x * log(temp)
    if y != 0.0:
        temp = x + r
        if temp > 0.0:
            v += y * log(temp)
    if z != 0.0:
        v -= z * atan(x * y / (z * r))
    return v


@cython.ufunc
@cython.cdivision
cdef api double prism_fzz(double x, double y, double z) nogil:
    """Evaluates the indefinite volume integral for the d**2/dz**2 * 1/r kernel.

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
    Can be used to compute other components by cycling the inputs
    """
    cdef:
        double v = 0.0
        double r

    if z != 0.0:
        r = sqrt(x * x + y * y + z * z)
        v = atan(x * y / (z * r))
    return v


@cython.ufunc
@cython.cdivision
cdef api double prism_fzx(double x, double y, double z) nogil:
    """Evaluates the indefinite volume integral for the d**2/(dz*dx) * 1/r kernel.

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
    Can be used to compute other components by cycling the inputs
    """
    cdef:
        double v = 0.0
        double r
    r = sqrt(x * x + y * y + z * z)
    v = y + r
    if v == 0.0:
        if y < 0:
            v = log(-2 * y)
    else:
        v = -log(v)
    return v


@cython.ufunc
@cython.cdivision
cdef api double prism_fzy(double x, double y, double z) nogil:
    """Evaluates the indefinite volume integral for the d**2/(dz*dy) * 1/r kernel.

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
    Can be used to compute other components by cycling the inputs
    """
    cdef:
        double v = 0.0
        double r, temp
    r = sqrt(x * x + y * y + z * z)
    v = x + r
    if v == 0.0:
        if x < 0:
            v = log(-2 * x)
    else:
        v = -log(v)
    return v


@cython.ufunc
@cython.cdivision
cdef api double prism_fzzz(double x, double y, double z) nogil:
    """Evaluates the indefinite volume integral for the d**3/(dz**3) * 1/r kernel.

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
    Can be used to compute other components by cycling the inputs
    """
    cdef:
        double v = 0.0
        double r, v1, v2
    r = sqrt(x * x + y * y + z * z)
    v1 = x * x + z * z
    v2 = y * y + z * z
    if v1 != 0.0:
        v += 1.0/v1
    if v2 != 0.0:
        v += 1.0/v2
    if r != 0.0:
        v *= x * y / r
    return v


@cython.ufunc
@cython.cdivision
cdef api double prism_fxxy(double x, double y, double z) nogil:
    """Evaluates the indefinite volume integral for the d**3/(dx**2 * dy) * 1/r kernel.

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
    Can be used to compute other components by cycling the inputs
    """
    cdef:
        double v = 0.0
        double r
    if x != 0.0:
        v = x * x + y * y
        r = sqrt(x * x + y * y + z * z)
        v = - x * z / (v * r)
    return v


@cython.ufunc
@cython.cdivision
cdef api double prism_fxxz(double x, double y, double z) nogil:
    """Evaluates the indefinite volume integral for the d**3/(dx**2 * dz) * 1/r kernel.

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
    Can be used to compute other components by cycling the inputs
    """
    cdef:
        double v = 0.0
        double r
    if x != 0.0:
        v = x * x + z * z
        r = sqrt(x * x + y * y + z * z)
        v = - x * y / (v * r)
    return v


@cython.ufunc
@cython.cdivision
cdef api double prism_fxyz(double x, double y, double z) nogil:
    """Evaluates the indefinite volume integral for the d**3/(dx * dy * dz) * 1/r kernel.

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
    Can be used to compute other components by cycling the inputs
    """
    cdef:
        double v = 0.0
        double r
    r = sqrt(x * x + y * y + z * z)
    if r != 0.0:
        v = 1.0/r
    return v