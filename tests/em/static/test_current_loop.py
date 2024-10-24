import pytest
import numpy as np
import numpy.testing as npt
from scipy.constants import mu_0

from geoana.em.static import CircularLoopWholeSpace, LineCurrentWholeSpace
from geoana.spatial import cylindrical_2_cartesian

METHODS = [
    'vector_potential', 'magnetic_field', 'magnetic_flux_density'
]

@pytest.fixture()
def circle_loop():

    radius = 5
    current = 1

    sigma = 1E-3
    mu = 4 * mu_0
    loop = CircularLoopWholeSpace(radius=radius, current=current, sigma=sigma, mu=mu)

    return loop

@pytest.fixture()
def xyz():
    nx, ny, nz = (32, 32, 32)
    x = np.linspace(-50, 50, nx)
    y = np.linspace(-50, 50, ny)
    z = np.linspace(-50, 50, nz)
    xyz = np.meshgrid(x, y, z)
    return xyz


@pytest.mark.parametrize('method', METHODS)
def test_broadcasting(circle_loop, xyz, method):

    func = getattr(circle_loop, method)

    out = func(xyz)
    assert out.shape == (*xyz[0].shape, 3)

    xyz = np.stack(xyz, axis=-1)
    out = func(xyz)
    assert out.shape == (*xyz.shape[:-1], 3)

    xyz = xyz.reshape((-1, 3))
    out = func(xyz)
    assert out.shape == (xyz.shape[0], 3)

    xyz = xyz.reshape((-1, 3))
    out = func(xyz)
    assert out.shape == (xyz.shape[0], 3)

    out = func(xyz[0])
    assert out.shape == (3,)


@pytest.mark.parametrize('orient', ['x', 'y', 'z'])
@pytest.mark.parametrize('method', METHODS)
def test_circular_loop(circle_loop, orient, method, xyz):
    circle_loop.orientation = orient

    # line segment approx.
    n_segments = 256
    rs = np.full(n_segments + 1, circle_loop.radius)
    thetas = np.linspace(0, 2 * np.pi, n_segments + 1)

    # defined as if orient = 'z'
    nodes = cylindrical_2_cartesian(np.c_[rs, thetas, np.zeros(n_segments + 1)])
    # Ensure it wraps.
    nodes[-1] = nodes[0]

    # cycle node for other orientations
    if orient == 'x':
        nodes = np.c_[nodes[:, 2], nodes[:, 0], nodes[:, 1]]
    elif orient == 'y':
        nodes = np.c_[nodes[:, 1], nodes[:, 2], nodes[:, 0]]

    loop_approx = LineCurrentWholeSpace(
        nodes,
        current=circle_loop.current,
        sigma=circle_loop.sigma,
        mu=circle_loop.mu
    )

    test = getattr(circle_loop, method)(xyz)
    other = getattr(loop_approx, method)(xyz)


    npt.assert_allclose(test, other, rtol=1E-3, atol=1E-20)