import numpy as np

__all__ = [
    "polyhedron_edge_log",
    "triangle_solid_angle",
    "face_edge_solid_angle",
    "tetrahedron_integrals",
    "tetrahedron_inside_fraction",
]


def _polyhedron_edge_log(x1, y1, z1, x2, y2, z2):
    """
    Evaluates the line integral of 1/r along a straight edge.

    This is the edge term of the closed-form volume integrals of polyhedra
    (Werner and Scheeres, 1996):
    :math:`L = \\ln\\frac{r_1 + r_2 + \\ell}{r_1 + r_2 - \\ell}`, for the edge
    whose end points are at (x1, y1, z1) and (x2, y2, z2) relative to the
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
    :math:`r_1 + r_2 - \\ell` cancels catastrophically near the edge and near
    its end points, so it is instead evaluated as
    :math:`\\frac{2(r_1 r_2 + \\mathbf{r}_1\\cdot\\mathbf{r}_2)}{r_1 + r_2 + \\ell}` when
    :math:`\\mathbf{r}_1\\cdot\\mathbf{r}_2 \\geq 0`, and as
    :math:`\\frac{2|\\mathbf{r}_1\\times\\mathbf{r}_2|^2}{(r_1 r_2 - \\mathbf{r}_1\\cdot\\mathbf{r}_2)(r_1 + r_2 + \\ell)}`
    otherwise, which are exact algebraically and free of cancellation.
    """
    x1, y1, z1, x2, y2, z2 = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (x1, y1, z1, x2, y2, z2))
    )
    n1 = np.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
    n2 = np.sqrt(x2 * x2 + y2 * y2 + z2 * z2)
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    length = np.sqrt(dx * dx + dy * dy + dz * dz)
    dot = x1 * x2 + y1 * y2 + z1 * z2
    s = n1 + n2
    cx = y1 * z2 - z1 * y2
    cy = z1 * x2 - x1 * z2
    cz = x1 * y2 - y1 * x2
    with np.errstate(divide="ignore", invalid="ignore"):
        diff = np.where(
            dot >= 0,
            2.0 * (n1 * n2 + dot) / (s + length),
            2.0 * (cx * cx + cy * cy + cz * cz) / ((n1 * n2 - dot) * (s + length)),
        )
    out = np.full(s.shape, np.inf)
    ok = diff > 0
    out[ok] = np.log((s[ok] + length[ok]) / diff[ok])
    return out


def _triangle_solid_angle(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    """
    Evaluates the signed solid angle subtended by a triangle.

    Uses the formula of van Oosterom and Strackee (1983) for the triangle whose
    vertices are at (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) relative to the
    observation point. The sign is positive when the vertices appear
    counterclockwise from the observation point.

    Parameters
    ----------
    x1, y1, z1, x2, y2, z2, x3, y3, z3 : (...) numpy.ndarray
        Vertex locations relative to the observation point.

    Returns
    -------
    (...) numpy.ndarray
        Solid angle in steradians, in :math:`[-2\\pi, 2\\pi]`. Zero when the
        observation point is in the triangle's plane, including on the triangle
        itself, where zero is the mean of the limits from either side.
    """
    x1, y1, z1, x2, y2, z2, x3, y3, z3 = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (x1, y1, z1, x2, y2, z2, x3, y3, z3))
    )
    n1 = np.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
    n2 = np.sqrt(x2 * x2 + y2 * y2 + z2 * z2)
    n3 = np.sqrt(x3 * x3 + y3 * y3 + z3 * z3)
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
    return np.where(num == 0, 0.0, 2.0 * np.arctan2(num, den))


def _face_edge_solid_angle(x1, y1, z1, x2, y2, z2, nx, ny, nz, h):
    """
    Evaluates the signed solid angle subtended by one edge of a planar face.

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
        normal, i.e. :math:`\\hat{n}\\cdot\\mathbf{r}` for any point
        :math:`\\mathbf{r}` on the face relative to the observation point.

    Returns
    -------
    (...) numpy.ndarray
        Solid angle in steradians, in :math:`[-\\pi, \\pi]`. Zero when ``h`` is
        zero, so the polygon's solid angle is zero on the polygon itself, the
        mean of its limits from either side.

    Notes
    -----
    With :math:`c = \\hat{n}\\cdot(\\mathbf{r}_1\\times\\mathbf{r}_2)`, the solid
    angle is
    :math:`2\\operatorname{atan2}(\\operatorname{sign}(h) c, r_1 r_2 + \\mathbf{r}_1\\cdot\\mathbf{r}_2 + |h|(r_1 + r_2))`
    (van Oosterom and Strackee, 1983, with one vertex at the foot of the
    perpendicular). Near the edge :math:`r_1 r_2 + \\mathbf{r}_1\\cdot\\mathbf{r}_2`
    cancels, so there it is evaluated as
    :math:`\\frac{|\\mathbf{r}_1\\times\\mathbf{r}_2|^2}{r_1 r_2 - \\mathbf{r}_1\\cdot\\mathbf{r}_2}`.
    """
    x1, y1, z1, x2, y2, z2, nx, ny, nz, h = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (x1, y1, z1, x2, y2, z2, nx, ny, nz, h))
    )
    n1 = np.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
    n2 = np.sqrt(x2 * x2 + y2 * y2 + z2 * z2)
    dot = x1 * x2 + y1 * y2 + z1 * z2
    cx = y1 * z2 - z1 * y2
    cy = z1 * x2 - x1 * z2
    cz = x1 * y2 - y1 * x2
    c = nx * cx + ny * cy + nz * cz
    with np.errstate(divide="ignore", invalid="ignore"):
        den = np.where(
            dot >= 0, n1 * n2 + dot, (cx * cx + cy * cy + cz * cz) / (n1 * n2 - dot)
        )
    sign = np.sign(h)
    out = 2.0 * np.arctan2(sign * c, den + np.abs(h) * (n1 + n2))
    return np.where(h == 0, 0.0, out)


try:
    from geoana.kernels._extensions.potential_field_polyhedron import (
        polyhedron_edge_log,
        triangle_solid_angle,
        face_edge_solid_angle,
    )
except ImportError:
    polyhedron_edge_log = _polyhedron_edge_log
    triangle_solid_angle = _triangle_solid_angle
    face_edge_solid_angle = _face_edge_solid_angle


def _canonical_face_normals(w):
    """Unit normals of triangles (..., 3 vertices, 3), wound as given.

    Each normal is computed from the triangle's vertices in lexicographic order,
    so any body with the same face, in either orientation, gets exactly the same
    normal up to sign, and their contributions for a shared face cancel exactly.

    Returns the normals (..., 3), the order (..., 3) that sorts each triangle's
    vertices lexicographically, and twice the triangles' areas (...).
    """
    n = np.cross(w[..., 1, :] - w[..., 0, :], w[..., 2, :] - w[..., 0, :])
    area2 = np.linalg.norm(n, axis=-1)
    order = np.lexsort((w[..., 2], w[..., 1], w[..., 0]), axis=-1)
    ws = np.take_along_axis(w, order[..., None], axis=-2)
    n_c = np.cross(ws[..., 1, :] - ws[..., 0, :], ws[..., 2, :] - ws[..., 0, :])
    with np.errstate(invalid="ignore", divide="ignore"):
        n_c = n_c / np.linalg.norm(n_c, axis=-1)[..., None]
    n = np.where((np.einsum("...i,...i->...", n_c, n) > 0)[..., None], n_c, -n_c)
    return n, order, area2


def _edge_dyads(t, n_a, n_b):
    """Edge dyads E = n_A ne_A^T + n_B ne_B^T and edge lengths.

    ``t`` (..., 3) runs from the edge's first end point to its second, face A
    traverses it in that direction and face B in the other; ``n_a`` and ``n_b``
    are their unit outward normals.
    """
    length = np.linalg.norm(t, axis=-1)
    t = t / length[..., None]
    # In-plane outward edge normals.
    ne_a = np.cross(t, n_a)
    ne_b = np.cross(-t, n_b)
    E = n_a[..., :, None] * ne_a[..., None, :] + n_b[..., :, None] * ne_b[..., None, :]
    return E, length


def _combine_newtonian(order, r_a, r_b, L, edge_lengths, E, h, omega, n, F):
    """V, grad V or the Hessian of V from the edge and face terms.

    Leading dimensions broadcast. ``r_a``, ``r_b`` (..., n_edges, 3) point to
    each edge's end points, ``L`` (..., n_edges) is its line integral,
    ``edge_lengths`` and ``E`` (..., n_edges[, 3, 3]) its length and dyad;
    ``h`` and ``omega`` (..., n_faces) are each face's height and solid angle,
    and ``n``, ``F`` (..., n_faces, 3[, 3]) its normal and dyad.
    """
    on_edge = np.isinf(L)
    if order < 2:
        # On an edge L is infinite, but it multiplies E . r_a, which vanishes
        # there as the distance d to the edge, and d ln(d) -> 0.
        L = np.where(on_edge, 0.0, L)
    elif np.any(on_edge):
        # The Hessian diverges as -2 E ln(d); keep the finite part, as Prism
        # does: ln(4 r1 r2) on the edge, ln(2 l) at its end points.
        r12 = np.linalg.norm(r_a, axis=-1) * np.linalg.norm(r_b, axis=-1)
        with np.errstate(divide="ignore"):
            finite = np.where(r12 > 0, np.log(4 * r12), np.log(2 * edge_lengths))
        L = np.where(on_edge, finite, L)
    if order == 0:
        er = np.einsum("...ei,...eij,...ej->...e", r_a, E, r_a)
        return 0.5 * (er * L).sum(axis=-1) - 0.5 * (h * h * omega).sum(axis=-1)
    if order == 1:
        return -np.einsum("...eij,...ej,...e->...i", E, r_a, L) + np.einsum(
            "...fi,...f,...f->...i", n, h, omega
        )
    if order == 2:
        return np.einsum("...eij,...e->...ij", E, L) - np.einsum("...fij,...f->...ij", F, omega)
    raise ValueError(f"order must be 0, 1 or 2, got {order}")


# Inside fractions this close to 0 or 1 are taken to be exactly 0 or 1.
_FRACTION_TOL = 1e-12


def _inside_fraction(omega):
    """Fraction of a small sphere inside a body from its faces' solid angles
    (..., n_faces): their sum over 4 pi, with roundoff snapped so points off
    the surface are exactly inside (1) or outside (0)."""
    frac = omega.sum(axis=-1) / (4.0 * np.pi)
    frac = np.where(np.abs(frac) < _FRACTION_TOL, 0.0, frac)
    return np.where(np.abs(frac - 1.0) < _FRACTION_TOL, 1.0, frac)


# Outward faces of a positively oriented tetrahedron, its edges, and for each
# edge the faces traversing it forwards and backwards.
_TET_FACES = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
_TET_EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
_TET_EDGE_FACES = np.array([
    [
        next(i for i, f in enumerate(_TET_FACES) if (a, b) in zip(f, np.roll(f, -1)))
        for a, b in (edge, edge[::-1])
    ]
    for edge in _TET_EDGES
])


def tetrahedron_integrals(vertices, xyz, order=0):
    r"""Evaluates the volume integrals of 1/r over tetrahedra in closed form.

    Computes :math:`V = \int_\Omega \frac{dV'}{|\mathbf{r} - \mathbf{r}'|}`, its
    gradient or its Hessian over each tetrahedron, with the conventions of
    :class:`geoana.shapes.BasePolyhedron`. The leading dimensions of
    ``vertices`` and ``xyz`` broadcast against each other, so this evaluates one
    tetrahedron at many points, many tetrahedra at one point, or both.

    Parameters
    ----------
    vertices : (..., 4, 3) array_like of float
        The four vertex locations of each tetrahedron, in any order.
    xyz : (..., 3) array_like of float
        Observation locations.
    order : {0, 1, 2}
        0 for :math:`V`, 1 for :math:`\nabla V` and 2 for
        :math:`\nabla\nabla V`.

    Returns
    -------
    (...) or (..., 3) or (..., 3, 3) numpy.ndarray
        With leading dimensions ``np.broadcast_shapes(vertices.shape[:-2],
        xyz.shape[:-1])``.

    Examples
    --------
    Many tetrahedra at one point:

    >>> import numpy as np
    >>> from geoana.kernels import tetrahedron_integrals
    >>> tets = np.random.default_rng(0).uniform(size=(100, 4, 3))
    >>> tetrahedron_integrals(tets, [0.5, 0.5, 2.0], order=1).shape
    (100, 3)

    Every tetrahedron at every one of 20 points:

    >>> xyz = np.random.default_rng(1).uniform(size=(20, 3))
    >>> tetrahedron_integrals(tets, xyz[:, None, :], order=2).shape
    (20, 100, 3, 3)
    """
    if order not in (0, 1, 2):
        raise ValueError(f"order must be 0, 1 or 2, got {order}")
    return _combine_newtonian(order, *_tetrahedron_terms(vertices, xyz))


def tetrahedron_inside_fraction(vertices, xyz):
    """Evaluates the fraction of a small sphere about each point inside tetrahedra.

    This is 1 inside a tetrahedron, 0 outside, and on its surface the solid
    angle it subtends from the point over :math:`4\\pi`: 1/2 on a face, and
    the dihedral angle over :math:`2\\pi` on an edge. It weights the
    magnetization in :math:`\\mathbf{B} = \\mu_0(\\mathbf{H} + f\\mathbf{M})`,
    which makes the normal component of B exact on a face (it is continuous
    there) and the tangential component the mean of its limits from either
    side, consistent with the Hessian from :func:`tetrahedron_integrals`. The
    fractions of tetrahedra that tile a region sum to that region's.

    Parameters
    ----------
    vertices : (..., 4, 3) array_like of float
        The four vertex locations of each tetrahedron, in any order.
    xyz : (..., 3) array_like of float
        Observation locations.

    Returns
    -------
    (...) numpy.ndarray
        With shape ``np.broadcast_shapes(vertices.shape[:-2], xyz.shape[:-1])``.

    Examples
    --------
    >>> from geoana.kernels import tetrahedron_inside_fraction
    >>> tet = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    >>> tetrahedron_inside_fraction(tet, [[0.1, 0.1, 0.1], [0.2, 0.2, 0.0], [2.0, 0.0, 0.0]])
    array([1. , 0.5, 0. ])
    """
    return _inside_fraction(_tetrahedron_terms(vertices, xyz)[6])


def _tetrahedron_terms(vertices, xyz):
    """Edge and face terms of tetrahedra (..., 4, 3) at points (..., 3), in the
    argument order of ``_combine_newtonian`` after ``order``."""
    v = np.array(vertices, dtype=float)
    xyz = np.asarray(xyz, dtype=float)
    if v.shape[-2:] != (4, 3):
        raise ValueError(f"vertices must have shape (..., 4, 3), got {v.shape}")
    if xyz.shape[-1] != 3:
        raise ValueError(f"xyz must have shape (..., 3), got {xyz.shape}")
    # Orient every tetrahedron positively.
    e1, e2, e3 = (v[..., k, :] - v[..., 0, :] for k in (1, 2, 3))
    volume = np.einsum("...i,...i->...", e1, np.cross(e2, e3))
    if np.any(volume == 0):
        raise ValueError("vertices include a tetrahedron with zero volume")
    v = np.where((volume < 0)[..., None, None], v[..., [0, 2, 1, 3], :], v)

    w = v[..., _TET_FACES, :]
    n, sort, _ = _canonical_face_normals(w)
    ref = np.take_along_axis(w, sort[..., :1, None], axis=-2)[..., 0, :]
    F = n[..., :, None] * n[..., None, :]
    a, b = _TET_EDGES[:, 0], _TET_EDGES[:, 1]
    E, length = _edge_dyads(
        v[..., b, :] - v[..., a, :], n[..., _TET_EDGE_FACES[:, 0], :], n[..., _TET_EDGE_FACES[:, 1], :]
    )

    x = xyz[..., None, :]
    r = v - x
    r_a, r_b = r[..., a, :], r[..., b, :]
    L = polyhedron_edge_log(*np.moveaxis(r_a, -1, 0), *np.moveaxis(r_b, -1, 0))
    h = np.einsum("...fi,...fi->...f", ref - x, n)
    r1 = r[..., _TET_FACES, :]
    r2 = np.roll(r1, -1, axis=-2)
    omega = face_edge_solid_angle(
        *np.moveaxis(r1, -1, 0),
        *np.moveaxis(r2, -1, 0),
        *np.moveaxis(n[..., None, :], -1, 0),
        h[..., None],
    ).sum(axis=-1)
    return r_a, r_b, L, length, E, h, omega, n, F
