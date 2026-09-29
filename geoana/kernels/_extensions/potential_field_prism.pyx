# cython: freethreading_compatible = True
# cython: language_level=3
# cython: embedsignature=True
cimport cython

from libc.math cimport sqrt, log, atan


cdef inline double _plus_r(double a, double b, double c, double r) nogil:
    """a + r, with r = sqrt(a**2 + b**2 + c**2), without cancellation for a < 0."""
    if a < 0.0:
        return (b * b + c * c) / (r - a)
    return a + r


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
        temp = _plus_r(z, x, y, r)
        if temp > 0.0:
            v -= x * y * log(temp)
        v += 0.5 * x * x * atan( y * z / (x * r))
    if y != 0.0 and z != 0.0:
        temp = _plus_r(x, y, z, r)
        if temp > 0.0:
            v -= y * z * log(temp)
        v += 0.5 * y * y * atan(z * x / (y * r))
    if z != 0.0 and x != 0.0:
        temp = _plus_r(y, x, z, r)
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
        temp = _plus_r(y, x, z, r)
        if temp > 0.0:
            v += x * log(temp)
    if y != 0.0:
        temp = _plus_r(x, y, z, r)
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
        double r, temp
    r = sqrt(x * x + y * y + z * z)
    if y < 0.0:
        # y + r cancels near the negative y axis; there it equals
        # (x**2 + z**2) / (r - y). On the axis the kernel diverges as
        # -2 ln(d) with the distance d to it; keep the finite part.
        temp = x * x + z * z
        if temp == 0.0:
            return log(-2 * y)
        return log((r - y) / temp)
    v = y + r
    if v == 0.0:
        return 0.0
    return -log(v)


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
    if x < 0.0:
        # x + r cancels near the negative x axis; there it equals
        # (y**2 + z**2) / (r - x). On the axis the kernel diverges as
        # -2 ln(d) with the distance d to it; keep the finite part.
        temp = y * y + z * z
        if temp == 0.0:
            return log(-2 * x)
        return log((r - x) / temp)
    v = x + r
    if v == 0.0:
        return 0.0
    return -log(v)


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