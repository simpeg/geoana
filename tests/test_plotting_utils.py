import numpy as np
from geoana.plotting_utils import plot2Ddata
from geoana.utils import ndgrid
from geoana.em.static import MagneticDipoleWholeSpace


def test_plot_2d_data():
    xyz = ndgrid(np.linspace(-1, 1, 5), np.array([0]), np.linspace(-1, 1, 5))
    location = np.r_[0., 0., 0.]
    orientation = np.r_[0., 0., 1.]
    moment = 1.
    dipole_object = MagneticDipoleWholeSpace(location=location, orientation=orientation, moment=moment)
    data = dipole_object.magnetic_flux_density(xyz)
    plot2Ddata(xyz[:, 0::2], data[:, 0::2], clim=np.array([1, 2]),
               vec=True, method='nearest', shade=True, figname='plot', dataloc=True)
