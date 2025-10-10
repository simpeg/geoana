import numpy as np
import pytest
from geoana.plotting_utils import plot2Ddata
from geoana.utils import ndgrid
from geoana.em.static import MagneticDipoleWholeSpace
from geoana.em.static import ElectrostaticSphere

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    matplotlib = None


@pytest.mark.skipif(matplotlib is None, reason="matplotlib not installed")
def test_plot_2d_data():
    xyz = np.array([np.linspace(-2, 2, 20), np.linspace(-2, 2, 20), np.linspace(-2, 2, 20)]).T
    location = np.r_[0., 0., 0.]
    orientation = np.r_[0., 0., 1.]
    moment = 1.
    dipole_object = MagneticDipoleWholeSpace(location=location, orientation=orientation, moment=moment)
    data1 = dipole_object.magnetic_flux_density(xyz)
    plot2Ddata(xyz[:, 0::2], data1[:, 0::2], clim=np.array([1, 2]), vec=True,
               method='nearest', shade=True, figname='plot', dataloc=True, contourOpts={'vmin': 1, 'vmax': 2, })

    with pytest.raises(Exception):
        plot2Ddata(xyz[:, 0::2], data1[:, 0::2], clim=np.array([1, 2]), vec=True,
                   method='nearest', contourOpts={'vmin': 3, 'vmax': 3, })

    sigma_sphere = 10. ** -1
    sigma_background = 10. ** -3
    radius = 1.0
    simulation = ElectrostaticSphere(location=None, sigma_sphere=sigma_sphere,
                                     sigma_background=sigma_background, radius=radius, primary_field=None)
    data2 = simulation.potential(xyz, field='total')
    plot2Ddata(xyz, data2, method='nearest', level=True, mask=np.full((20, ), True))

    xyz = ndgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20), np.array([0]))
    data3 = dipole_object.vector_potential(xyz)
    plot2Ddata(xyz[:, 0:2], data3[:, 0:2], vec=True, mask=np.full((400, ), True),
               contourOpts={'vmin': 0.8, 'vmax': 20, })


