import numpy as np
import numpy.testing as npt

import pytest
from scipy.constants import mu_0
from scipy.special import roots_legendre

from geoana.em.static import (
    MagneticPrism,
    MagneticDipoleWholeSpace,
    MagneticPolyhedron,
    MagneticTetrahedron,
)
from geoana.utils import append_ndim


class TestMagneticAccuracy():
    x, y, z = np.mgrid[-100:100:20j, -100:100:20j, -100:100:20j]
    xyz = np.stack((x, y, z), axis=-1).reshape((-1, 3))
    dx = 0.1
    prism = MagneticPrism(dx * np.r_[-1, -1, -1], dx * np.r_[1, 1, 1], magnetization=[-1, 2, -0.5])
    m_mag = np.linalg.norm(prism.magnetization)
    m_unit = prism.magnetization/m_mag
    dipole = MagneticDipoleWholeSpace(
        location=[0, 0, 0], moment=m_mag, orientation=m_unit
    )

    quad_points, quad_weights = roots_legendre(5)
    quad_points = (prism.max_location - prism.min_location)[:, None] * (quad_points + 1) / 2 + prism.min_location[:,
                                                                                               None]
    quad_points = np.stack(np.meshgrid(*quad_points, indexing='ij'), axis=-1)
    quad_wx, quad_wy, quad_wz = quad_weights * (prism.max_location - prism.min_location)[:, None] / 2
    quad_wx = quad_wx[:, None, None]
    quad_wy = quad_wy[None, :, None]
    quad_wz = quad_wz[None, None, :]

    quad_xyzs = xyz - quad_points[..., None, :]

    @pytest.mark.parametrize(
        'method,rtol',
        [
            ('magnetic_field', 1E-7),
            ('magnetic_flux_density', 1E-7),
         ]
    )
    def test_accuracy(self, method, rtol):
        test_prism = getattr(self.prism, method)(self.xyz)

        wx = append_ndim(self.quad_wx, test_prism.ndim)
        wy = append_ndim(self.quad_wy, test_prism.ndim)
        wz = append_ndim(self.quad_wz, test_prism.ndim)

        test_quad = getattr(self.dipole, method)(self.quad_xyzs)
        test_quad *= wx
        test_quad *= wy
        test_quad *= wz
        test_quad = np.sum(test_quad, axis=(0, 1, 2))

        atol = rtol * (test_prism.max() - test_prism.min())
        npt.assert_allclose(test_quad, test_prism, atol=atol)

def _box(lo, hi):
    V = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    F = []
    for a, b, c, d in quads:
        F += [(a, b, c), (a, c, d)]
    return V, np.array(F)


@pytest.mark.parametrize(
    "method", ["scalar_potential", "magnetic_field", "magnetic_flux_density"]
)
def test_magnetic_polyhedron_matches_prism(method):
    lo, hi = np.r_[-1.0, -2.0, -3.0], np.r_[2.0, 1.5, 0.5]
    m = [-1.0, 2.0, -0.5]
    V, F = _box(lo, hi)
    poly = MagneticPolyhedron(V, F, magnetization=m)
    prism = MagneticPrism(lo, hi, magnetization=m)
    rng = np.random.default_rng(1)
    out = rng.uniform(-8, 8, (400, 3))
    out = out[~np.all((out >= lo - 0.3) & (out <= hi + 0.3), axis=1)]
    ins = rng.uniform(lo + 0.1, hi - 0.1, (100, 3))
    for xyz in (out, ins):
        a = getattr(poly, method)(xyz)
        b = getattr(prism, method)(xyz)
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12 * np.max(np.abs(b)))


@pytest.mark.parametrize("method", ["scalar_potential", "magnetic_field", "magnetic_flux_density"])
def test_magnetic_polyhedron_on_boundary(method):
    # The scalar potential is continuous, so it is defined on the surface too:
    # vertices, edge midpoints and face centers (on the face diagonals). The
    # fields follow Prism's conventions there.
    lo, hi = np.r_[-1.0, -2.0, -3.0], np.r_[2.0, 1.5, 0.5]
    m = [-1.0, 2.0, -0.5]
    V, F = _box(lo, hi)
    poly = MagneticPolyhedron(V, F, magnetization=m)
    prism = MagneticPrism(lo, hi, magnetization=m)
    c = 0.5 * (lo + hi)
    xyz = np.vstack([
        V,
        [[hi[0], c[1], hi[2]], [lo[0], lo[1], c[2]], [c[0], hi[1], lo[2]]],
        [[c[0], c[1], hi[2]], [lo[0], c[1], c[2]], [c[0], hi[1], c[2]]],
    ])
    a = getattr(poly, method)(xyz)
    b = getattr(prism, method)(xyz)
    assert np.all(np.isfinite(a))
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12 * np.max(np.abs(b)))


def test_magnetic_tetrahedron_potential_continuous():
    v = np.array([[0, 0, 0], [3, 0.5, 0], [1, 2, 0.2], [0.7, 0.4, 1.9]])
    tet = MagneticTetrahedron(v, magnetization=[0.3, -0.2, 1.0])
    # A face centroid, an edge midpoint, a point a third along an edge and a vertex.
    xyz = np.array([v[[1, 2, 3]].mean(axis=0), 0.5 * (v[1] + v[3]), v[0] + (v[2] - v[0]) / 3, v[3]])
    step = np.r_[0.31, -0.52, 0.79] / np.linalg.norm([0.31, -0.52, 0.79])
    on = tet.scalar_potential(xyz)
    for eps in [1e-9, 1e-12, 1e-15]:
        for sign in (1, -1):
            near = tet.scalar_potential(xyz + sign * eps * step)
            np.testing.assert_allclose(near, on, rtol=0, atol=1e-6 * np.max(np.abs(on)))


def test_magnetic_tetrahedron_far_field_is_dipole():
    v = np.array([[0, 0, 0], [3, 0.5, 0], [1, 2, 0.2], [0.7, 0.4, 1.9]])
    m = np.r_[0.3, -0.2, 1.0]
    tet = MagneticTetrahedron(v, magnetization=m)
    far = np.array([[300.0, 600.0, -900.0], [-1500.0, 30.0, 600.0]])
    moment = tet.moment
    r = far - tet.location
    rn = np.linalg.norm(r, axis=-1, keepdims=True)
    H = (3 * r * (r @ moment)[:, None] / rn**2 - moment) / (4 * np.pi * rn**3)
    np.testing.assert_allclose(tet.magnetic_field(far), H, rtol=1e-5, atol=1e-5 * np.abs(H).max())


def _check_flux_density_on_face(body, x0, normal, eps=1e-9, exactly_on=True):
    """On a face, the normal component of B is continuous, so it must equal its
    limits from either side; the tangential component jumps, and must be their
    mean. A point that is on the face only to within roundoff may instead take
    either side's limit."""
    B_on = body.magnetic_flux_density(x0[None])[0]
    B_in = body.magnetic_flux_density((x0 - eps * normal)[None])[0]
    B_out = body.magnetic_flux_density((x0 + eps * normal)[None])[0]
    scale = np.abs(B_in).max()
    np.testing.assert_allclose(B_on @ normal, B_in @ normal, rtol=0, atol=1e-7 * scale)
    np.testing.assert_allclose(B_on @ normal, B_out @ normal, rtol=0, atol=1e-7 * scale)
    allowed = [0.5 * (B_in + B_out)] if exactly_on else [0.5 * (B_in + B_out), B_in, B_out]
    assert min(np.abs(B_on - b).max() for b in allowed) < 1e-7 * scale
    # The limits really do differ tangentially.
    assert np.abs(B_in - B_out).max() > 0.1 * scale


def test_prism_flux_density_on_face():
    lo, hi = np.r_[-1.0, -2.0, -3.0], np.r_[2.0, 1.5, 0.5]
    prism = MagneticPrism(lo, hi, magnetization=[-1.0, 2.0, -0.5])
    _check_flux_density_on_face(prism, np.r_[0.37, -0.61, hi[2]], np.r_[0.0, 0.0, 1.0])
    _check_flux_density_on_face(prism, np.r_[lo[0], 0.2, -1.1], np.r_[-1.0, 0.0, 0.0])


def test_prism_flux_density_inside_fraction():
    # B - mu_0 H is mu_0 M weighted by the fraction of a small sphere inside the
    # prism: 1/2 on a face, 1/4 on an edge, 1/8 at a corner.
    lo, hi = np.r_[-1.0, -2.0, -3.0], np.r_[2.0, 1.5, 0.5]
    m = np.r_[-1.0, 2.0, -0.5]
    prism = MagneticPrism(lo, hi, magnetization=m)
    xyz = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, hi[2]], [hi[0], 0.0, hi[2]], hi, [5.0, 0.0, 0.0]])
    frac = np.r_[1.0, 0.5, 0.25, 0.125, 0.0]
    M = prism.magnetic_flux_density(xyz) / mu_0 - prism.magnetic_field(xyz)
    np.testing.assert_allclose(M, frac[:, None] * m, atol=1e-14)


def test_tetrahedron_flux_density_on_face():
    v = np.array([[0, 0, 0], [3, 0.5, 0], [1, 2, 0.2], [0.7, 0.4, 1.9]])
    tet = MagneticTetrahedron(v, magnetization=[0.3, -0.2, 1.0])
    normal = np.cross(v[2] - v[1], v[3] - v[1])
    normal /= np.linalg.norm(normal)
    normal *= np.sign(normal @ (v[1] - v[0]))
    # The centroid of this skewed face is on it only to within roundoff.
    _check_flux_density_on_face(tet, v[[1, 2, 3]].mean(axis=0), normal, exactly_on=False)
    # A face in the plane z = 0 contains its points exactly.
    flat = MagneticTetrahedron(
        [[0, 0, 0], [3, 0.5, 0], [1, 2, 0], [0.7, 0.4, 1.9]], magnetization=[0.3, -0.2, 1.0]
    )
    _check_flux_density_on_face(flat, np.r_[1.2, 0.8, 0.0], np.r_[0.0, 0.0, -1.0])


def _box_tetrahedra(lo, hi):
    """Six tetrahedra that tile an axis-aligned box (Kuhn triangulation)."""
    V = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    tets = []
    for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        idx = [0]
        for axis in perm:
            idx.append(idx[-1] + (4, 2, 1)[axis])
        tets.append(V[idx])
    return tets


def test_tetrahedra_flux_density_tile_prism():
    # Each point on a face shared by two tetrahedra is half inside each, so the
    # tetrahedra sum to the prism there, as well as on the prism's own surface.
    lo, hi = np.r_[-1.0, -2.0, -3.0], np.r_[2.0, 1.5, 0.5]
    m = [-1.0, 2.0, -0.5]
    prism = MagneticPrism(lo, hi, magnetization=m)
    tets = [MagneticTetrahedron(v, magnetization=m) for v in _box_tetrahedra(lo, hi)]
    c = 0.5 * (lo + hi)
    xyz = np.array([
        lo + 0.25 * (hi - lo),
        lo + np.r_[0.6, 0.3, 0.3] * (hi - lo),
        lo + np.r_[0.7, 0.7, 0.2] * (hi - lo),
        [c[0], c[1], hi[2]],
        [0.37, -0.61, hi[2]],
        [hi[0], c[1], hi[2]],
        hi,
    ])
    a = sum(t.magnetic_flux_density(xyz) for t in tets)
    b = prism.magnetic_flux_density(xyz)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-13 * np.abs(b).max())

