"""
======================================================
Shapes (:mod:`geoana.shapes`)
======================================================
.. currentmodule:: geoana.shapes

The ``geoana.shapes`` module contains the basic geometric building blocks for
three dimensional objects.

Shape Classes
=============
.. autosummary::
  :toctree: generated/

  BasePrism
  BasePolyhedron
  BaseTetrahedron
"""
import numpy as np

from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components

from geoana.kernels.potential_field_polyhedron import (
    polyhedron_edge_log,
    face_edge_solid_angle,
    _canonical_face_normals,
    _combine_newtonian,
    _edge_dyads,
    _inside_fraction,
)

__all__ = [
    "BasePrism",
    "BasePolyhedron",
    "BaseTetrahedron",
]

# Faces whose unit normals differ by less than this are treated as coplanar.
_COPLANAR_TOL = 1e-12


class BasePrism:
    """Class for basic geometry of a prism.

    The ``BasePrism`` class is used to define and validate basic geometry of an axis
    aligned prism in three dimensions.

    Parameters
    ----------
    min_location : (3,) numpy.ndarray of float
        Minimum location triple of the axis aligned prism
    max_location : (3,) numpy.ndarray of float
        Maximum location triple of the axis aligned prism
    """

    def __init__(self, min_location, max_location, **kwargs):
        super().__init__(**kwargs)
        self.min_location = min_location
        self.max_location = max_location
        if np.any(self.max_location <= self.min_location):
            raise ValueError("Max location must be strictly greater than the minimum location")

    @property
    def min_location(self):
        """Location of the point mass.

        Returns
        -------
        (3) numpy.ndarray of float
            Location of the point mass in meters.
        """
        return self._min_location

    @min_location.setter
    def min_location(self, vec):

        try:
            vec = np.asarray(vec, dtype=float)
        except:
            raise TypeError(f"location must be array_like of float, got {type(vec)}")

        vec = np.squeeze(vec)
        if vec.shape != (3,):
            raise ValueError(
                f"location must be array_like with shape (3,), got {vec.shape}"
            )

        self._min_location = vec

    @property
    def max_location(self):
        """Location of the point mass.

        Returns
        -------
        (3) numpy.ndarray of float
            Location of the point mass in meters.
        """
        return self._max_location

    @max_location.setter
    def max_location(self, vec):

        try:
            vec = np.asarray(vec, dtype=float)
        except:
            raise TypeError(f"location must be array_like of float, got {type(vec)}")

        vec = np.squeeze(vec)
        if vec.shape != (3,):
            raise ValueError(
                f"location must be array_like with shape (3,), got {vec.shape}"
            )

        self._max_location = vec

    @property
    def volume(self):
        """ The volume of the prism

        Returns
        -------
        float
        """
        return np.prod(self.max_location - self.min_location)

    @property
    def location(self):
        """ The center of the prism

        Returns
        -------
        (3,) numpy.ndarray of float
        """
        return 0.5 * (self.min_location + self.max_location)

    def _eval_def_int(self, func, x, y, z, cycle=0):
        "evaluate a definite integral (func) over the prism at x, y, z locations"

        x_min, y_min, z_min = self.min_location
        x_max, y_max, z_max = self.max_location

        x_min = x_min - x
        y_min = y_min - y
        z_min = z_min - z

        x_max = x_max - x
        y_max = y_max - y
        z_max = z_max - z

        for i in range(cycle):
            x_min, y_min, z_min = y_min, z_min, x_min
            x_max, y_max, z_max = y_max, z_max, x_max

        v000 = func(x_min, y_min, z_min)
        v001 = func(x_min, y_min, z_max)
        v010 = func(x_min, y_max, z_min)
        v011 = func(x_min, y_max, z_max)
        v100 = func(x_max, y_min, z_min)
        v101 = func(x_max, y_min, z_max)
        v110 = func(x_max, y_max, z_min)
        v111 = func(x_max, y_max, z_max)

        val = (v111 - v110 - v101 + v100 - v011 + v010 + v001 - v000)
        return val


class BasePolyhedron:
    r"""Class for basic geometry of a closed polyhedron with triangular faces.

    The ``BasePolyhedron`` class defines and validates a closed, consistently
    oriented triangulated surface, and evaluates the volume integrals of the
    Newtonian kernel :math:`1/|\mathbf{r} - \mathbf{r}'|` over the enclosed body
    in closed form, following Werner and Scheeres (1996):

    .. math::

        V(\mathbf{r}) &= \int_\Omega \frac{dV'}{|\mathbf{r} - \mathbf{r}'|}
            = \frac{1}{2}\sum_e \mathbf{r}_e \cdot \mathbf{E}_e \cdot \mathbf{r}_e L_e
            - \frac{1}{2}\sum_f \mathbf{r}_f \cdot \mathbf{F}_f \cdot \mathbf{r}_f \omega_f

        \nabla V &= -\sum_e \mathbf{E}_e \cdot \mathbf{r}_e L_e
            + \sum_f \mathbf{F}_f \cdot \mathbf{r}_f \omega_f

        \nabla\nabla V &= \sum_e \mathbf{E}_e L_e - \sum_f \mathbf{F}_f \omega_f

    where :math:`\mathbf{r}_e` and :math:`\mathbf{r}_f` point from the
    observation point to any point on edge :math:`e` or face :math:`f`,
    :math:`\mathbf{F}_f = \hat{n}_f \hat{n}_f^T` is the face dyad,
    :math:`\mathbf{E}_e = \hat{n}_A \hat{n}^A_e{}^T + \hat{n}_B \hat{n}^B_e{}^T` is
    the edge dyad built from the two faces sharing the edge and their in-plane
    edge normals, :math:`L_e = \ln\frac{r_1 + r_2 + \ell_e}{r_1 + r_2 - \ell_e}` is
    the edge's line integral, and :math:`\omega_f` is the signed solid angle the
    face subtends (van Oosterom and Strackee, 1983). The solid angles sum to
    :math:`4\pi` inside the body and to zero outside, so
    :math:`\nabla^2 V = -4\pi` inside and zero outside.

    Adjacent coplanar triangles are merged into planar polygonal faces, whose
    solid angles are summed edge by edge over their boundaries, so the
    diagonals of a triangulated flat face play no part.

    :math:`V` and :math:`\nabla V` are continuous everywhere. On the surface,
    :math:`\nabla\nabla V` takes the same values as
    :class:`geoana.gravity.Prism`: on a face, the mean of its limits from either
    side, and on an edge, where it diverges as :math:`-2\mathbf{E}_e\ln d` with
    the distance :math:`d` to the edge, the finite part left by dropping that
    term, :math:`L_e = \ln(4 r_1 r_2)`, or :math:`\ln(2\ell_e)` at the edge's
    end points. These conventions are additive, so bodies that tile a region sum
    to that region's values on their shared boundaries as well.

    Parameters
    ----------
    vertices : (n_vertices, 3) array_like of float
        Vertex locations.
    faces : (n_faces, 3) array_like of int
        Vertex indices of each triangular face. Every edge must be shared by
        exactly two faces traversed in opposite directions, i.e. the surface is
        closed and consistently oriented. The orientation may be either inward
        or outward; it is made outward on construction.

    References
    ----------
    Werner, R. A., and D. J. Scheeres (1996), Exterior gravitation of a
    polyhedron derived and compared with harmonic and mascon gravitation
    representations of asteroid 4769 Castalia, Celestial Mechanics and Dynamical
    Astronomy, 65, 313-344.

    van Oosterom, A., and J. Strackee (1983), The solid angle of a plane
    triangle, IEEE Transactions on Biomedical Engineering, BME-30(2), 125-126.
    """

    def __init__(self, vertices, faces, **kwargs):
        super().__init__(**kwargs)
        vertices = np.asarray(vertices, dtype=float)
        faces = np.asarray(faces)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(
                f"vertices must have shape (n_vertices, 3), got {vertices.shape}"
            )
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"faces must have shape (n_faces, 3), got {faces.shape}")
        if not np.issubdtype(faces.dtype, np.integer):
            raise TypeError(f"faces must be integer indices, got {faces.dtype}")
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError("faces reference a vertex index out of range")
        self._vertices = vertices
        self._faces = faces.astype(np.intp)
        self._build_edges()
        if self.volume < 0:
            self._faces = self._faces[:, ::-1].copy()
            self._build_edges()
        if self.volume <= 0:
            raise ValueError("polyhedron has zero volume")
        self._build_dyads()

    def _build_edges(self):
        faces = self._faces
        directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        owner = np.tile(np.arange(len(faces)), 3)
        key = {}
        for (a, b), f in zip(map(tuple, directed), owner):
            if (a, b) in key:
                raise ValueError(
                    f"edge {a}->{b} is traversed twice in the same direction; "
                    "faces are not consistently oriented"
                )
            key[(a, b)] = f
        edges, face_a, face_b = [], [], []
        for (a, b), f in key.items():
            if a < b:
                if (b, a) not in key:
                    raise ValueError(f"edge {a}-{b} belongs to only one face; surface is not closed")
                edges.append((a, b))
                face_a.append(f)
                face_b.append(key[(b, a)])
        if len(edges) * 2 != len(directed):
            raise ValueError("surface is not closed")
        self._edges = np.asarray(edges, dtype=np.intp)
        # face_a traverses the edge a -> b, face_b traverses it b -> a.
        self._edge_faces = np.asarray([face_a, face_b], dtype=np.intp).T

    def _build_dyads(self):
        v = self._vertices
        f = self._faces
        n, order, area2 = _canonical_face_normals(v[f])
        if np.any(area2 == 0):
            raise ValueError("polyhedron has a degenerate (zero-area) face")
        face_ref = np.take_along_axis(f, order, axis=-1)[:, 0]

        n_a = n[self._edge_faces[:, 0]]
        n_b = n[self._edge_faces[:, 1]]
        # An edge between coplanar faces (e.g. the diagonal of a triangulated
        # flat face) has a vanishing dyad, so it contributes nothing. Its dyad
        # is only roundoff, which would make 0 * inf on the edge itself.
        coplanar = np.linalg.norm(np.cross(n_a, n_b), axis=-1) < _COPLANAR_TOL
        coplanar &= np.einsum("ij,ij->i", n_a, n_b) > 0

        # Merge coplanar faces into planar polygons.
        n_faces = len(f)
        ef = self._edge_faces[coplanar]
        graph = coo_array(
            (np.ones(len(ef)), (ef[:, 0], ef[:, 1])), shape=(n_faces, n_faces)
        )
        n_groups, group = connected_components(graph, directed=False)
        counts = np.bincount(group, minlength=n_groups)
        area_n = np.zeros((n_groups, 3))
        np.add.at(area_n, group, n * area2[:, None])
        n_g = area_n / np.linalg.norm(area_n, axis=-1)[:, None]
        single = counts[group] == 1
        n_g[group[single]] = n[single]
        # Reference vertex of each polygon: its lexicographically first vertex.
        rank = np.empty(len(v), dtype=np.intp)
        rank[np.lexsort((v[:, 2], v[:, 1], v[:, 0]))] = np.arange(len(v))
        ref_rank = np.full(n_groups, len(v))
        np.minimum.at(ref_rank, group, rank[face_ref])
        self._face_ref = np.argsort(rank)[ref_rank]
        self._face_normals = n_g
        self._face_dyads = n_g[:, :, None] * n_g[:, None, :]

        # Directed boundary edges of the polygons, grouped by polygon.
        directed = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        owner = group[np.tile(np.arange(n_faces), 3)]
        n_v = len(v)
        key = lambda e: np.minimum(e[:, 0], e[:, 1]) * n_v + np.maximum(e[:, 0], e[:, 1])
        boundary = ~np.isin(key(directed), key(self._edges[coplanar]))
        directed, owner = directed[boundary], owner[boundary]
        sort = np.argsort(owner, kind="stable")
        self._face_edges = directed[sort]
        self._face_edge_group = owner[sort]
        self._face_edge_starts = np.searchsorted(owner[sort], np.arange(n_groups))

        self._edges = self._edges[~coplanar]
        self._edge_faces = self._edge_faces[~coplanar]
        n_a, n_b = n_a[~coplanar], n_b[~coplanar]

        a, b = self._edges[:, 0], self._edges[:, 1]
        # Face A traverses the edge a->b, face B b->a.
        self._edge_dyads, self._edge_lengths = _edge_dyads(v[b] - v[a], n_a, n_b)

    @property
    def vertices(self):
        """Vertex locations.

        Returns
        -------
        (n_vertices, 3) numpy.ndarray of float
        """
        return self._vertices

    @property
    def faces(self):
        """Vertex indices of each triangular face, ordered so normals point outward.

        Returns
        -------
        (n_faces, 3) numpy.ndarray of int
        """
        return self._faces

    @property
    def volume(self):
        """The volume of the polyhedron.

        Returns
        -------
        float
        """
        v = self._vertices[self._faces]
        return float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0)

    @property
    def location(self):
        """The centroid of the polyhedron.

        Returns
        -------
        (3,) numpy.ndarray of float
        """
        v = self._vertices[self._faces]
        w = np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2]))
        return (w[:, None] * v.sum(axis=1)).sum(axis=0) / (4.0 * w.sum())

    def _kernel_terms(self, xyz):
        """Edge and face terms at the (n, 3) observation points.

        Returns r_e (n, n_edges, 3) and r_b (n, n_edges, 3), the vectors to each
        edge's two end points, L_e (n, n_edges), h_f (n, n_faces), the distance
        along each face normal to the face's plane, and omega_f (n, n_faces).
        """
        r = self._vertices[None, :, :] - xyz[:, None, :]

        a, b = self._edges[:, 0], self._edges[:, 1]
        r_e, r_b = r[:, a], r[:, b]
        L = polyhedron_edge_log(
            r_e[..., 0], r_e[..., 1], r_e[..., 2], r_b[..., 0], r_b[..., 1], r_b[..., 2]
        )

        n = self._face_normals
        h = np.einsum("nfi,fi->nf", r[:, self._face_ref], n)
        grp = self._face_edge_group
        r1, r2 = r[:, self._face_edges[:, 0]], r[:, self._face_edges[:, 1]]
        n_e = n[grp]
        omega = face_edge_solid_angle(
            r1[..., 0], r1[..., 1], r1[..., 2],
            r2[..., 0], r2[..., 1], r2[..., 2],
            n_e[:, 0], n_e[:, 1], n_e[:, 2],
            h[:, grp],
        )
        omega = np.add.reduceat(omega, self._face_edge_starts, axis=-1)
        return r_e, r_b, L, h, omega

    def _newtonian_integrals(self, xyz, order):
        """V, grad V and/or the Hessian of V at observation points.

        ``order`` is 0 (V), 1 (grad V) or 2 (Hessian). ``xyz`` is (..., 3).
        """
        shape = xyz.shape[:-1]
        pts = xyz.reshape(-1, 3)
        r_e, r_b, L, h, omega = self._kernel_terms(pts)
        out = _combine_newtonian(
            order, r_e, r_b, L, self._edge_lengths, self._edge_dyads,
            h, omega, self._face_normals, self._face_dyads,
        )
        return out.reshape(*shape, *out.shape[1:])

    def _inside_fraction(self, xyz):
        """Fraction of a small sphere about each (..., 3) point inside the body.

        One inside, zero outside, and on the surface the solid angle of the body
        seen from the point over 4 pi, e.g. 1/2 on a face.
        """
        shape = xyz.shape[:-1]
        omega = self._kernel_terms(xyz.reshape(-1, 3))[-1]
        return _inside_fraction(omega).reshape(shape)


class BaseTetrahedron(BasePolyhedron):
    """Class for basic geometry of a tetrahedron.

    Parameters
    ----------
    vertices : (4, 3) array_like of float
        The four vertex locations, in any order.
    """

    def __init__(self, vertices, **kwargs):
        vertices = np.asarray(vertices, dtype=float)
        if vertices.shape != (4, 3):
            raise ValueError(f"vertices must have shape (4, 3), got {vertices.shape}")
        faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
        super().__init__(vertices, faces, **kwargs)
