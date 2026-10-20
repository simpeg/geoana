# cython: freethreading_compatible = True
# cython: language_level=3
# cython: embedsignature=True
cimport cython

from libc.math cimport sqrt, log, atan2, INFINITY

@cython.cdivision
@cython.ufunc
cdef api double polyhedron_edge_log(
    double x1, double y1, double z1,
    double x2, double y2, double z2,
) nogil:
    """Evaluates the line integral of 1/r along a straight edge.

    This is the edge term of the closed-form volume integrals of polyhedra
    (Werner and Scheeres, 1996): ln((r1 + r2 + l) / (r1 + r2 - l)), for the
    edge whose end points are at (x1, y1, z1) and (x2, y2, z2) relative to the
    observation point.

    Parameters
    ----------
    x1, y1, z1, x2, y2, z2 : (...) numpy.ndarray
        Locations of the edge's two end points relative to the observation
        point.

    Returns
    -------
    (...) numpy.ndarray
        Infinite where the observation point lies on the edge.

    Notes
    -----
    r1 + r2 - l cancels catastrophically near the edge and near its end
    points, so it is instead evaluated as 2 (r1 r2 + r1 . r2) / (r1 + r2 + l)
    when r1 . r2 >= 0, and as 2 |r1 x r2|^2 / ((r1 r2 - r1 . r2)(r1 + r2 + l))
    otherwise, which are exact algebraically and free of cancellation.
    """
    cdef:
        double n1 = sqrt(x1 * x1 + y1 * y1 + z1 * z1)
        double n2 = sqrt(x2 * x2 + y2 * y2 + z2 * z2)
        double dx = x2 - x1, dy = y2 - y1, dz = z2 - z1
        double length = sqrt(dx * dx + dy * dy + dz * dz)
        double dot = x1 * x2 + y1 * y2 + z1 * z2
        double s = n1 + n2
        double cx, cy, cz, num, den, diff
    if n1 == 0.0 or n2 == 0.0:
        # At an end point.
        return INFINITY
    # Choose the numerator and denominator, then divide once: compilers may
    # evaluate both branches, and the untaken one can be 0 / 0.
    if dot >= 0.0:
        num = n1 * n2 + dot
        den = 1.0
    else:
        cx = y1 * z2 - z1 * y2
        cy = z1 * x2 - x1 * z2
        cz = x1 * y2 - y1 * x2
        num = cx * cx + cy * cy + cz * cz
        den = n1 * n2 - dot
    diff = 2.0 * num / (den * (s + length))
    if diff > 0.0:
        return log((s + length) / diff)
    return INFINITY

@cython.cdivision
@cython.ufunc
cdef api double triangle_solid_angle(
    double x1, double y1, double z1,
    double x2, double y2, double z2,
    double x3, double y3, double z3,
) nogil:
    """Evaluates the signed solid angle subtended by a triangle.

    Uses the formula of van Oosterom and Strackee (1983) for the triangle whose
    vertices are given relative to the observation point. The sign is positive
    when the vertices appear counterclockwise from the observation point.

    Parameters
    ----------
    x1, y1, z1, x2, y2, z2, x3, y3, z3 : (...) numpy.ndarray
        Vertex locations relative to the observation point.

    Returns
    -------
    (...) numpy.ndarray
        Solid angle in steradians, in [-2 pi, 2 pi]. Zero when the observation
        point is in the triangle's plane, including on the triangle itself,
        where zero is the mean of the limits from either side.
    """
    cdef:
        double n1 = sqrt(x1 * x1 + y1 * y1 + z1 * z1)
        double n2 = sqrt(x2 * x2 + y2 * y2 + z2 * z2)
        double n3 = sqrt(x3 * x3 + y3 * y3 + z3 * z3)
        double num, den
    num = (
        x1 * (y2 * z3 - z2 * y3)
        + y1 * (z2 * x3 - x2 * z3)
        + z1 * (x2 * y3 - y2 * x3)
    )
    den = (
        n1 * n2 * n3
        + n1 * (x2 * x3 + y2 * y3 + z2 * z3)
        + n2 * (x3 * x1 + y3 * y1 + z3 * z1)
        + n3 * (x1 * x2 + y1 * y2 + z1 * z2)
    )
    if num == 0.0:
        return 0.0
    return 2.0 * atan2(num, den)

@cython.cdivision
@cython.ufunc
cdef api double face_edge_solid_angle(
    double x1, double y1, double z1,
    double x2, double y2, double z2,
    double nx, double ny, double nz,
    double h,
) nogil:
    """Evaluates the signed solid angle subtended by one edge of a planar face.

    This is the solid angle of the triangle formed by the edge and the foot of
    the perpendicular from the observation point to the face's plane. Summed
    over the directed boundary edges of a planar polygon, it gives the solid
    angle the polygon subtends, without the polygon's internal diagonals, and
    two faces that share an edge in opposite directions have exactly opposite
    terms for it.

    Parameters
    ----------
    x1, y1, z1, x2, y2, z2 : (...) numpy.ndarray
        Locations of the edge's start and end points relative to the
        observation point.
    nx, ny, nz : (...) numpy.ndarray
        Unit normal of the face; the edge runs counterclockwise about it.
    h : (...) numpy.ndarray
        Distance from the observation point to the face's plane along the
        normal, i.e. n . r for any point r on the face relative to the
        observation point.

    Returns
    -------
    (...) numpy.ndarray
        Solid angle in steradians, in [-pi, pi]. Zero when h is zero, so the
        polygon's solid angle is zero on the polygon itself, the mean of its
        limits from either side.

    Notes
    -----
    With c = n . (r1 x r2), the solid angle is
    2 atan2(sign(h) c, r1 r2 + r1 . r2 + |h| (r1 + r2)) (van Oosterom and
    Strackee, 1983, with one vertex at the foot of the perpendicular). Near the
    edge r1 r2 + r1 . r2 cancels, so there it is evaluated as
    |r1 x r2|^2 / (r1 r2 - r1 . r2).
    """
    cdef:
        double n1 = sqrt(x1 * x1 + y1 * y1 + z1 * z1)
        double n2 = sqrt(x2 * x2 + y2 * y2 + z2 * z2)
        double dot = x1 * x2 + y1 * y2 + z1 * z2
        double cx = y1 * z2 - z1 * y2
        double cy = z1 * x2 - x1 * z2
        double cz = x1 * y2 - y1 * x2
        double c = nx * cx + ny * cy + nz * cz
        double num, den
    if h == 0.0 or n1 == 0.0 or n2 == 0.0:
        # In the plane, or at an end point (where c is zero).
        return 0.0
    # Choose the numerator and denominator, then divide once: compilers may
    # evaluate both branches, and the untaken one can be 0 / 0.
    if dot >= 0.0:
        num = n1 * n2 + dot
        den = 1.0
    else:
        num = cx * cx + cy * cy + cz * cz
        den = n1 * n2 - dot
    den = num / den
    if h > 0.0:
        return 2.0 * atan2(c, den + h * (n1 + n2))
    return 2.0 * atan2(-c, den - h * (n1 + n2))
