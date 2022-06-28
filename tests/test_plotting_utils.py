import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from geoana.plotting_utils import plot2Ddata
from geoana.utils import ndgrid
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from geoana.em.static import MagneticDipoleWholeSpace


def test_plot_2d_data():
    xyz = ndgrid(np.linspace(-1, 1, 20), np.array([0]), np.linspace(-1, 1, 20))
    location = np.r_[0., 0., 0.]
    orientation = np.r_[0., 0., 1.]
    moment = 1.
    dipole_object = MagneticDipoleWholeSpace(location=location, orientation=orientation, moment=moment)
    f = dipole_object.magnetic_flux_density(xyz)
    cont_test, ax_test = plot2Ddata(xyz[:, 0::2], f[:, 0::2], clim=np.array([1, 2]))

    vmin, vmax = 1, 2
    ax = plt.subplot(111)
    xmin, xmax = xyz[:, 0].min(), xyz[:, 0].max()
    ymin, ymax = xyz[:, 1].min(), xyz[:, 1].max()
    x = np.linspace(xmin, xmax, 100)
    y = np.linspace(ymin, ymax, 100)
    X, Y = np.meshgrid(x, y)
    xy = np.c_[X.flatten(), Y.flatten()]

    f = LinearNDInterpolator(xyz[:, :2], f)
    data = f(xy)
    data = data.reshape(X.shape)

    levels = np.linspace(vmin, vmax, 11)
    contourOpts = {'levels': levels, 'zorder': 3}
    cont = ax.contour(X, Y, data, **contourOpts)

    np.testing.assert_equal(cont_test, cont)
    np.testing.assert_equal(ax_test, ax)






