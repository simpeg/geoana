import numpy as np
import numpy.testing as npt
import pytest

import geoana.kernels.potential_field_polyhedron as pp
try:
    from numba import njit
except ImportError:
    njit = None


class TestCompiledVsNumpy():
    rng = np.random.default_rng(42)
    tri = rng.uniform(-10, 10, size=(9, 2000))
    edge = rng.uniform(-10, 10, size=(6, 2000))

    def test_assert_using_compiled(self):
        assert pp._polyhedron_edge_log is not pp.polyhedron_edge_log
        assert pp._triangle_solid_angle is not pp.triangle_solid_angle
        assert pp._face_edge_solid_angle is not pp.face_edge_solid_angle

    def test_edge_log(self):
        v0 = pp._polyhedron_edge_log(*self.edge)
        v1 = pp.polyhedron_edge_log(*self.edge)
        npt.assert_allclose(v0, v1)

    @pytest.mark.parametrize("func", [pp._polyhedron_edge_log, pp.polyhedron_edge_log])
    def test_edge_log_on_edge(self, func):
        # On the edge r1 + r2 == length and the line integral diverges.
        assert np.isinf(func(1.0, 0.0, 0.0, -2.0, 0.0, 0.0))
        # At an end point too.
        assert np.isinf(func(0.0, 0.0, 0.0, -2.0, 0.0, 0.0))
        # Collinear but beyond the end is finite: the integral of 1/x from 1 to 2.
        npt.assert_allclose(func(1.0, 0.0, 0.0, 2.0, 0.0, 0.0), np.log(2.0))

    @pytest.mark.parametrize("func", [pp._triangle_solid_angle, pp.triangle_solid_angle])
    def test_solid_angle_in_plane(self, func):
        # In the triangle's plane, on or off the triangle, the solid angle is zero
        # (on the triangle, the mean of its limits from either side).
        npt.assert_equal(func(-1, -1, 0, 2, 0, 0, 0, 2, 0), 0.0)
        npt.assert_equal(func(-1, -1, 0, 0, 2, 0, 2, 0, 0), 0.0)
        npt.assert_equal(func(1, 1, 0, 2, 1, 0, 1, 2, 0), 0.0)

    def test_solid_angle(self):
        v0 = pp._triangle_solid_angle(*self.tri)
        v1 = pp.triangle_solid_angle(*self.tri)
        npt.assert_allclose(v0, v1)

    def test_face_edge_solid_angle(self):
        args = _face_edge_args(self.rng)
        v0 = pp._face_edge_solid_angle(*args)
        v1 = pp.face_edge_solid_angle(*args)
        npt.assert_allclose(v0, v1, atol=1e-14)


def _face_edge_args(rng, n=2000):
    """Random triangles relative to the observation point, split into their
    directed edges 0->1, 1->2 and 2->0, each with the triangle's unit normal and
    height. Every returned array has shape (3, n), one row per edge."""
    r = rng.uniform(-10, 10, size=(3, n, 3))
    normal = np.cross(r[1] - r[0], r[2] - r[0])
    normal /= np.linalg.norm(normal, axis=-1)[:, None]
    h = np.einsum("ij,ij->i", normal, r[0])
    start, end = r, np.roll(r, -1, axis=0)
    normal = np.broadcast_to(normal, (3, n, 3))
    return (*np.moveaxis(start, -1, 0), *np.moveaxis(end, -1, 0), *np.moveaxis(normal, -1, 0),
            np.broadcast_to(h, (3, n)))


class TestKernelValues():

    @pytest.mark.parametrize("func", [pp._face_edge_solid_angle, pp.face_edge_solid_angle])
    def test_face_edge_terms_sum_to_triangle(self, func):
        rng = np.random.default_rng(3)
        args = _face_edge_args(rng)
        x1, y1, z1, x2, y2, z2 = args[:6]
        tri = (x1[0], y1[0], z1[0], x1[1], y1[1], z1[1], x1[2], y1[2], z1[2])
        npt.assert_allclose(func(*args).sum(axis=0), pp.triangle_solid_angle(*tri), atol=1e-12)

    @pytest.mark.parametrize("func", [pp._face_edge_solid_angle, pp.face_edge_solid_angle])
    def test_face_edge_antisymmetric(self, func):
        # Faces sharing an edge traverse it in opposite directions; their terms
        # must cancel exactly, including right next to the edge.
        rng = np.random.default_rng(4)
        args = _face_edge_args(rng)
        swapped = args[3:6] + args[0:3] + args[6:]
        npt.assert_array_equal(func(*swapped), -func(*args))
        # Next to the edge from (-1, 0, 0) to (1, 0, 0) in the plane z = 0.
        d = 1e-13
        near = (-1.0, d, d, 1.0, d, d, 0.0, 0.0, 1.0, d)
        swapped = near[3:6] + near[0:3] + near[6:]
        npt.assert_array_equal(func(*swapped), -func(*near))

    @pytest.mark.parametrize("func", [pp._face_edge_solid_angle, pp.face_edge_solid_angle])
    def test_face_edge_in_plane(self, func):
        npt.assert_equal(func(1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0), 0.0)
        # Just off the plane, the edge subtends its in-plane angle, here pi / 2,
        # negative on the side the normal points to.
        npt.assert_allclose(func(1.0, -1.0, -1e-9, 1.0, 1.0, -1e-9, 0.0, 0.0, 1.0, -1e-9), -np.pi / 2)
        npt.assert_allclose(func(1.0, -1.0, 1e-9, 1.0, 1.0, 1e-9, 0.0, 0.0, 1.0, 1e-9), np.pi / 2)

    def test_solid_angle_octant(self):
        # The unit triangle on the axes subtends one eighth of the sphere.
        npt.assert_allclose(
            pp.triangle_solid_angle(1, 0, 0, 0, 1, 0, 0, 0, 1), 4 * np.pi / 8
        )
        npt.assert_allclose(
            pp.triangle_solid_angle(1, 0, 0, 0, 0, 1, 0, 1, 0), -4 * np.pi / 8
        )

    @pytest.mark.parametrize("func", [pp._polyhedron_edge_log, pp.polyhedron_edge_log])
    def test_edge_log_is_line_integral(self, func):
        # A point on the perpendicular bisector of an edge of length 2 at distance d:
        # the integral of 1/r along the edge is 2 asinh(1/d). The small distances
        # check that r1 + r2 - length is not lost to cancellation near the edge.
        d = np.array([1e-150, 1e-15, 1e-10, 1e-7, 0.1, 1.0, 10.0])
        npt.assert_allclose(func(-1.0, d, 0.0, 1.0, d, 0.0), 2 * np.arcsinh(1 / d), rtol=1e-14)
        # Off the bisector: the line integral is asinh(b/d) - asinh(a/d).
        a, b = -0.3, 1.7
        npt.assert_allclose(
            func(a, d, 0.0, b, d, 0.0), np.arcsinh(b / d) - np.arcsinh(a / d), rtol=1e-14
        )
        # Next to an end point: the line integral is asinh(1 / d).
        npt.assert_allclose(func(0.0, d, 0.0, 1.0, d, 0.0), np.arcsinh(1 / d), rtol=1e-14)


@pytest.mark.skipif(njit is None, reason="Numba is not installed.")
class TestNumbaCallable():

    def test_edge_log(self):
        @njit
        def f(*args):
            return pp.polyhedron_edge_log(*args)
        args = (-1.0, 0.5, 0.0, 1.0, 0.5, 0.0)
        npt.assert_allclose(f(*args), pp._polyhedron_edge_log(*args))

    def test_solid_angle(self):
        @njit
        def f(*args):
            return pp.triangle_solid_angle(*args)
        args = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        npt.assert_allclose(f(*args), pp._triangle_solid_angle(*args))

    def test_face_edge_solid_angle(self):
        @njit
        def f(*args):
            return pp.face_edge_solid_angle(*args)
        args = (1.0, -1.0, -0.5, 1.0, 1.0, -0.5, 0.0, 0.0, 1.0, -0.5)
        npt.assert_allclose(f(*args), pp._face_edge_solid_angle(*args))


def _kuhn_grid(n):
    """Tetrahedra tiling the cube [0, n]^3: six per unit cell."""
    cells = np.stack(np.meshgrid(*[np.arange(float(n))] * 3, indexing="ij"), -1).reshape(-1, 3)
    corner = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    paths = []
    for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        idx = [0]
        for axis in perm:
            idx.append(idx[-1] + (4, 2, 1)[axis])
        paths.append(idx)
    return (cells[:, None, None, :] + corner[np.array(paths)][None]).reshape(-1, 4, 3)


class TestTetrahedronIntegrals():
    rng = np.random.default_rng(7)
    tets = rng.uniform(-2, 2, size=(30, 4, 3))
    xyz = np.vstack([
        rng.uniform(-4, 4, size=(10, 3)),
        tets[0],                                   # vertices of the first
        tets[0, [1, 2, 3]].mean(axis=0)[None],     # a face centroid
        0.5 * (tets[0, 0] + tets[0, 2])[None],     # an edge midpoint
    ])

    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_shapes(self, order):
        tail = (3,) * order
        assert pp.tetrahedron_integrals(self.tets[0], self.xyz, order).shape == (len(self.xyz), *tail)
        assert pp.tetrahedron_integrals(self.tets, self.xyz[0], order).shape == (len(self.tets), *tail)
        both = pp.tetrahedron_integrals(self.tets, self.xyz[:, None, :], order)
        assert both.shape == (len(self.xyz), len(self.tets), *tail)
        # Broadcasting either way gives the same values.
        one_tet = pp.tetrahedron_integrals(self.tets[3], self.xyz, order)
        one_pt = pp.tetrahedron_integrals(self.tets, self.xyz[5], order)
        npt.assert_array_equal(both[:, 3], one_tet)
        npt.assert_array_equal(both[5], one_pt)

    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_matches_tetrahedron(self, order):
        from geoana.shapes import BaseTetrahedron
        got = pp.tetrahedron_integrals(self.tets, self.xyz[:, None, :], order)
        for i, v in enumerate(self.tets):
            want = BaseTetrahedron(v)._newtonian_integrals(self.xyz, order)
            npt.assert_allclose(got[:, i], want, rtol=1e-12, atol=1e-12 * np.abs(want).max())

    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_vertex_order(self, order):
        a = pp.tetrahedron_integrals(self.tets, self.xyz[:, None, :], order)
        b = pp.tetrahedron_integrals(self.tets[:, [3, 1, 0, 2]], self.xyz[:, None, :], order)
        npt.assert_allclose(a, b, rtol=1e-12, atol=1e-12 * np.abs(a).max())

    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_mesh_sums_to_prism(self, order):
        # Many tetrahedra at once, observed on the mesh's surface, at a node, on
        # shared edges and faces inside it, and outside.
        from geoana.gravity import Prism
        from scipy.constants import G
        tets = _kuhn_grid(4)
        xyz = np.array([
            [1.5, 2.0, 4.0], [4.0, 4.0, 4.0], [2.0, 1.0, 3.0], [1.5, 1.5, 1.5],
            [2.25, 1.5, 1.5], [0.7, 0.3, 2.3], [6.0, -1.0, 2.0],
        ])
        got = G * pp.tetrahedron_integrals(tets, xyz[:, None, :], order).sum(axis=1)
        prism = Prism([0, 0, 0], [4, 4, 4], rho=1.0)
        want = getattr(prism, ["gravitational_potential", "gravitational_field", "gravitational_gradient"][order])(xyz)
        npt.assert_allclose(got, want, rtol=0, atol=1e-13 * np.abs(want).max())

    def test_inside_fraction_values(self):
        # The corner tetrahedron of the unit cube: its right-angled edges have a
        # dihedral angle of pi / 2, its right-angled corner an eighth of a sphere.
        tet = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        xyz = np.array([
            [0.1, 0.1, 0.1],   # inside
            [0.2, 0.2, 0.0],   # on a face
            [0.5, 0.0, 0.0],   # on a right-angled edge
            [0.0, 0.0, 0.0],   # at the right-angled corner
            [2.0, 0.0, 0.0],   # outside
            [0.25, 0.25, 0.5],  # on the slanted face (exactly, in binary)
        ])
        npt.assert_allclose(
            pp.tetrahedron_inside_fraction(tet, xyz), [1.0, 0.5, 0.25, 0.125, 0.0, 0.5], atol=1e-14
        )

    def test_inside_fraction_matches_polyhedron(self):
        from geoana.shapes import BaseTetrahedron
        got = pp.tetrahedron_inside_fraction(self.tets, self.xyz[:, None, :])
        assert got.shape == (len(self.xyz), len(self.tets))
        for i, v in enumerate(self.tets):
            npt.assert_allclose(got[:, i], BaseTetrahedron(v)._inside_fraction(self.xyz), atol=1e-14)

    def test_inside_fraction_mesh_sums_to_cube(self):
        # Fractions of tetrahedra tiling a cube sum to the cube's everywhere:
        # 1 inside (including on shared nodes, edges and faces), 1/2 on the
        # cube's faces, 1/4 on its edges, 1/8 at its corners, 0 outside.
        tets = _kuhn_grid(4)
        xyz = np.array([
            [2.0, 1.0, 3.0], [1.5, 1.5, 1.5], [2.25, 1.5, 1.5], [0.7, 0.3, 2.3],
            [1.5, 2.0, 4.0], [0.0, 1.0, 2.0], [4.0, 4.0, 2.5], [4.0, 4.0, 4.0], [6.0, -1.0, 2.0],
        ])
        got = pp.tetrahedron_inside_fraction(tets, xyz[:, None, :]).sum(axis=1)
        npt.assert_allclose(got, [1, 1, 1, 1, 0.5, 0.5, 0.25, 0.125, 0], atol=1e-13)

    def test_errors(self):
        with pytest.raises(ValueError):
            pp.tetrahedron_integrals(self.tets[:, :3], self.xyz)
        with pytest.raises(ValueError):
            pp.tetrahedron_integrals(self.tets, self.xyz[:, :2])
        with pytest.raises(ValueError):
            pp.tetrahedron_integrals(self.tets, self.xyz[0], order=3)
        flat = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        with pytest.raises(ValueError):
            pp.tetrahedron_integrals(flat, self.xyz[0])
