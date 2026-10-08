import pytest
import numpy as np
import numpy.testing as npt
from scipy.special import roots_legendre

from geoana import gravity
from geoana.utils import append_ndim

METHODS = [
    "gravitational_potential",
    "gravitational_field",
    "gravitational_gradient",
]

class TestPointMass:

    def test_defaults(self):
        pm = gravity.PointMass()
        assert pm.mass == 1
        assert np.all(pm.location == np.r_[0., 0., 0.])

    def test_errors(self):
        pm = gravity.PointMass(mass=1.0, location=None)
        with pytest.raises(TypeError):
            pm.mass = "string"
        with pytest.raises(ValueError):
            pm.location = [0, 1, 2, 3]
        with pytest.raises(ValueError):
            pm.location = [[0, 0, 1, 4], [0, 1, 0, 3]]
        with pytest.raises(TypeError):
            pm.location = ["string"]

    @pytest.mark.parametrize("method", METHODS)
    def test_correct(self, method, sympy_grav_point, grav_point_params):
        mass = grav_point_params["mass"]
        grav_obj = gravity.PointMass(
            mass=mass
        )
        x = np.linspace(-20., 20., 50)
        y = np.linspace(-30., 30., 50)
        z = np.linspace(-40., 40., 50)
        xyz = np.meshgrid(x, y, z)

        func = getattr(grav_obj, method)

        out = func(xyz)

        verify = sympy_grav_point[method](*xyz)

        np.testing.assert_allclose(out, verify)


class TestSphere:

    def test_defaults(self):
        radius = 1.0
        rho = 1.0
        s = gravity.Sphere(radius, rho)
        assert s.rho == 1
        assert s.radius == 1
        assert s.mass == 4 / 3 * np.pi * s.radius ** 3 * s.rho
        assert np.all(s.location == np.r_[0., 0., 0.])

    def test_errors(self):
        s = gravity.Sphere(rho=1.0, radius=1.0, location=None)
        with pytest.raises(TypeError):
            s.mass = "string"
        with pytest.raises(ValueError):
            s.radius = -1
        with pytest.raises(ValueError):
            s.location = [0, 1, 2, 3, 4]
        with pytest.raises(ValueError):
            s.location = [[0, 0, 1, 4], [0, 1, 0, 3]]
        with pytest.raises(TypeError):
            s.location = ["string"]

    @pytest.mark.parametrize("method", METHODS)
    def test_correct(self, method, sympy_grav_sphere, grav_point_params):
        mass = grav_point_params["mass"]
        radius = grav_point_params["radius"]
        vol = 4/3 * np.pi * radius**3
        rho = mass / vol

        grav_obj = gravity.Sphere(
            radius=radius,
            rho=rho,
        )
        x = np.linspace(-2., 2., 50)
        y = np.linspace(-3., 3., 50)
        z = np.linspace(-4., 4., 50)
        xyz = np.meshgrid(x, y, z)

        func = getattr(grav_obj, method)

        out = func(xyz)

        verify = sympy_grav_sphere[method](*xyz)

        np.testing.assert_allclose(out, verify)


class TestGravityAccuracy():
    x, y, z = np.mgrid[-100:100:20j, -100:100:20j, -100:100:20j]
    xyz = np.stack((x, y, z), axis=-1).reshape((-1, 3))
    dx = 0.1
    prism = gravity.Prism(dx * np.r_[-1, -1, -1], dx * np.r_[1, 1, 1], rho=2)

    quad_points, quad_weights = roots_legendre(5)
    quad_points = (prism.max_location - prism.min_location)[:, None] * (quad_points + 1) / 2 + prism.min_location[:,
                                                                                               None]
    quad_points = np.stack(np.meshgrid(*quad_points, indexing='ij'), axis=-1)
    quad_wx, quad_wy, quad_wz = quad_weights * (prism.max_location - prism.min_location)[:, None] / 2
    quad_wx = quad_wx[:, None, None]
    quad_wy = quad_wy[None, :, None]
    quad_wz = quad_wz[None, None, :]

    pm = gravity.PointMass(mass=prism.rho, location=[0, 0, 0])

    quad_xyzs = xyz - quad_points[..., None, :]

    @pytest.mark.parametrize(
        'method,rtol',
        [
            ('gravitational_potential', 1E-7),
            ('gravitational_field', 1E-7),
            ('gravitational_gradient', 1E-7),
         ]
    )
    def test_accuracy(self, method, rtol):
        test_prism = getattr(self.prism, method)(self.xyz)

        wx = append_ndim(self.quad_wx, test_prism.ndim)
        wy = append_ndim(self.quad_wy, test_prism.ndim)
        wz = append_ndim(self.quad_wz, test_prism.ndim)

        test_quad = getattr(self.pm, method)(self.quad_xyzs)
        test_quad *= wx
        test_quad *= wy
        test_quad *= wz
        test_quad = np.sum(test_quad, axis=(0, 1, 2))

        atol = rtol * (test_prism.max() - test_prism.min())
        npt.assert_allclose(test_quad, test_prism, atol=atol)

def _box_polyhedron(lo, hi):
    """Vertices and outward triangular faces of an axis-aligned box."""
    V = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    F = []
    for a, b, c, d in quads:
        F += [(a, b, c), (a, c, d)]
    return V, np.array(F)


def _box_tetrahedra(lo, hi):
    """Six tetrahedra that tile an axis-aligned box (Kuhn triangulation)."""
    V = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    # Paths from vertex 0 (min corner) to vertex 7 (max corner) along the axes.
    bit = {0: 4, 1: 2, 2: 1}
    tets = []
    for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        idx = [0]
        for axis in perm:
            idx.append(idx[-1] + bit[axis])
        tets.append(V[idx])
    return tets


def _box_boundary_points(lo, hi):
    """Points on the surface of a box: its vertices, edge midpoints, face
    centers (which lie on the diagonals of the triangulated faces) and
    off-center face points."""
    V = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    c = 0.5 * (lo + hi)
    pts = [V]
    for axis in range(3):
        for side in (lo, hi):
            face = c.copy()
            face[axis] = side[axis]
            pts.append(face)
            off = lo + np.r_[0.37, 0.61, 0.23] * (hi - lo)
            off[axis] = side[axis]
            pts.append(off)
            for other in (lo, hi):
                edge = c.copy()
                edge[axis] = side[axis]
                edge[(axis + 1) % 3] = other[(axis + 1) % 3]
                pts.append(edge)
    return np.vstack(pts)


# Edge, face and vertex points of a skewed tetrahedron, with a direction to
# step across them.
_TET = np.array([[0, 0, 0], [3, 0.5, 0], [1, 2, 0.2], [0.7, 0.4, 1.9]])
_TET_BOUNDARY = np.array([
    _TET[[1, 2, 3]].mean(axis=0),
    0.5 * (_TET[1] + _TET[3]),
    _TET[0] + (_TET[2] - _TET[0]) / 3,
    _TET[3],
])
_STEP = np.r_[0.31, -0.52, 0.79] / np.linalg.norm([0.31, -0.52, 0.79])


class TestPolyhedron:
    lo = np.r_[-1.0, -2.0, -3.0]
    hi = np.r_[2.0, 1.5, 0.5]
    rng = np.random.default_rng(0)
    outside = rng.uniform(-8, 8, (400, 3))
    outside = outside[~np.all((outside >= lo - 0.3) & (outside <= hi + 0.3), axis=1)]
    inside = rng.uniform(lo + 0.1, hi - 0.1, (100, 3))
    boundary = _box_boundary_points(lo, hi)
    # The box's main diagonal is an edge of every tetrahedron in its Kuhn
    # triangulation, and the other points lie on faces shared by two of them.
    shared = np.array([
        lo + 0.25 * (hi - lo),
        lo + 0.5 * (hi - lo),
        lo + np.r_[0.6, 0.3, 0.3] * (hi - lo),
        lo + np.r_[0.7, 0.7, 0.2] * (hi - lo),
    ])

    def test_geometry(self):
        V, F = _box_polyhedron(self.lo, self.hi)
        p = gravity.Polyhedron(V, F, rho=2.0)
        npt.assert_allclose(p.volume, np.prod(self.hi - self.lo))
        npt.assert_allclose(p.location, 0.5 * (self.lo + self.hi))
        npt.assert_allclose(p.mass, 2.0 * p.volume)

    def test_orientation_is_normalized(self):
        V, F = _box_polyhedron(self.lo, self.hi)
        inward = gravity.Polyhedron(V, F[:, ::-1])
        outward = gravity.Polyhedron(V, F)
        npt.assert_allclose(
            inward.gravitational_field(self.outside), outward.gravitational_field(self.outside)
        )

    def test_errors(self):
        V, F = _box_polyhedron(self.lo, self.hi)
        with pytest.raises(ValueError):
            gravity.Polyhedron(V, F[:-1])  # not closed
        bad = F.copy()
        bad[0] = bad[0][::-1]
        with pytest.raises(ValueError):
            gravity.Polyhedron(V, bad)  # inconsistently oriented
        with pytest.raises(TypeError):
            gravity.Polyhedron(V, F, rho="string")

    @pytest.mark.parametrize("method", METHODS)
    def test_box_matches_prism(self, method):
        V, F = _box_polyhedron(self.lo, self.hi)
        poly = gravity.Polyhedron(V, F, rho=2.5)
        prism = gravity.Prism(self.lo, self.hi, rho=2.5)
        for xyz in (self.outside, self.inside):
            a = getattr(poly, method)(xyz)
            b = getattr(prism, method)(xyz)
            npt.assert_allclose(a, b, rtol=1e-10, atol=1e-12 * np.max(np.abs(b)))

    @pytest.mark.parametrize("method", METHODS)
    def test_tetrahedra_tile_prism(self, method):
        prism = gravity.Prism(self.lo, self.hi, rho=1.5)
        tets = [gravity.Tetrahedron(v, rho=1.5) for v in _box_tetrahedra(self.lo, self.hi)]
        npt.assert_allclose(sum(t.volume for t in tets), prism.volume)
        a = sum(getattr(t, method)(self.outside) for t in tets)
        b = getattr(prism, method)(self.outside)
        npt.assert_allclose(a, b, rtol=1e-10, atol=1e-12 * np.max(np.abs(b)))

    @pytest.mark.parametrize("method", METHODS)
    def test_boundary_matches_prism(self, method):
        # The potential and field are continuous, so they are defined on the
        # surface, including on edges and vertices where the edge terms diverge.
        # The gradient follows Prism's conventions there.
        V, F = _box_polyhedron(self.lo, self.hi)
        poly = gravity.Polyhedron(V, F, rho=2.5)
        prism = gravity.Prism(self.lo, self.hi, rho=2.5)
        a = getattr(poly, method)(self.boundary)
        b = getattr(prism, method)(self.boundary)
        assert np.all(np.isfinite(a))
        npt.assert_allclose(a, b, rtol=1e-10, atol=1e-12 * np.max(np.abs(b)))

    @pytest.mark.parametrize("method", METHODS)
    @pytest.mark.parametrize("offset", [0.0, 1e-12, -1e-12])
    def test_tetrahedra_tile_prism_on_shared_boundaries(self, method, offset):
        # Subtracting the tetrahedra from the prism leaves nothing, on and right
        # next to every boundary, shared or not.
        prism = gravity.Prism(self.lo, self.hi, rho=1.5)
        tets = [gravity.Tetrahedron(v, rho=1.5) for v in _box_tetrahedra(self.lo, self.hi)]
        xyz = np.vstack([self.shared, self.boundary]) + offset * _STEP
        a = sum(getattr(t, method)(xyz) for t in tets)
        b = getattr(prism, method)(xyz)
        npt.assert_allclose(a, b, rtol=0, atol=1e-13 * np.max(np.abs(b)))

    @pytest.mark.parametrize("method", METHODS)
    def test_split_tetrahedron_sums_to_parent(self, method):
        # Four tetrahedra joining the faces of a skewed tetrahedron to an interior
        # point: on the parent's boundary and on the shared internal faces and
        # edges, the parts sum to the whole.
        p = np.r_[0.3, 0.2, 0.25, 0.25] @ _TET
        parts = [gravity.Tetrahedron(np.vstack([np.delete(_TET, i, axis=0), p])) for i in range(4)]
        whole = gravity.Tetrahedron(_TET)
        xyz = np.vstack([
            _TET_BOUNDARY,
            p,
            0.5 * (p + _TET[1]),
            (p + _TET[0] + _TET[2]) / 3,
        ])
        for offset in (0.0, 1e-12, -1e-12):
            x = xyz + offset * _STEP
            a = sum(getattr(t, method)(x) for t in parts)
            b = getattr(whole, method)(x)
            npt.assert_allclose(a, b, rtol=0, atol=1e-12 * np.max(np.abs(b)))

    @pytest.mark.parametrize("method", METHODS[:2])
    def test_continuous_across_boundary(self, method):
        tet = gravity.Tetrahedron(_TET, rho=3.0)
        f = getattr(tet, method)
        on = f(_TET_BOUNDARY)
        scale = np.max(np.abs(on))
        for eps in [1e-9, 1e-12, 1e-15]:
            for sign in (1, -1):
                near = f(_TET_BOUNDARY + sign * eps * _STEP)
                npt.assert_allclose(near, on, rtol=0, atol=1e-6 * scale)

    def test_far_field_is_point_mass(self):
        v = np.array([[0, 0, 0], [3, 0.5, 0], [1, 2, 0.2], [0.7, 0.4, 1.9]])
        tet = gravity.Tetrahedron(v, rho=3.0)
        pm = gravity.PointMass(mass=tet.mass, location=tet.location)
        # About 300 body sizes away: the quadrupole term is below 1e-5 of the
        # monopole, and both this kernel and Prism are still precise there (the
        # closed forms lose precision by cancellation past ~1e3 body sizes).
        far = np.array([[300.0, 600.0, -900.0], [-1500.0, 30.0, 600.0]])
        g_pm = pm.gravitational_field(far)
        npt.assert_allclose(tet.gravitational_field(far), g_pm, rtol=1e-5, atol=1e-5 * np.abs(g_pm).max())

    def test_laplace(self):
        V, F = _box_polyhedron(self.lo, self.hi)
        p = gravity.Polyhedron(V, F, rho=2.0)
        from scipy.constants import G
        tr_out = np.trace(p.gravitational_gradient(self.outside), axis1=-2, axis2=-1)
        tr_in = np.trace(p.gravitational_gradient(self.inside), axis1=-2, axis2=-1)
        npt.assert_allclose(tr_out, 0.0, atol=1e-12 * G * 2.0)
        npt.assert_allclose(tr_in, -4 * np.pi * G * 2.0, rtol=1e-10)
