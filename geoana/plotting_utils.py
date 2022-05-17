"""
=================================================
Plotting Utilities (:mod:`geoana.plotting_utils`)
=================================================
.. currentmodule:: geoana.plotting_utils

The ``geoana.plotting_utils`` module provides some functions for plotting
data computed with geoana.

.. autosummary::
  :toctree: generated/

  plot2Ddata

"""

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import matplotlib.pyplot as plt
from matplotlib import colors
import warnings


def plot2Ddata(
    xyz,
    data,
    vec=False,
    nx=100,
    ny=100,
    ax=None,
    mask=None,
    level=False,
    figname=None,
    ncontour=10,
    dataloc=False,
    contourOpts={},
    levelOpts={},
    streamplotOpts={},
    scale="linear",
    clim=None,
    method="linear",
    shade=False,
    shade_ncontour=100,
    shade_azimuth=-45.0,
    shade_angle_altitude=45.0,
    shadeOpts={},
):
    """
    Take unstructured xy points, interpolate, then plot in 2D

    Parameters
    ----------
    xyz : (n, dim) numpy.ndarray
        Data locations. Only the first two columns are used if an (n, 3)
        array is entered.
    data : (n) or (n, 2) np.ndarray
        Data values. Either scalar or 2D vector.
    vec : bool
        Default = ``False``. Plot vector data. If ``False``, the ``data`` input argument is 
        scalar-valued. If ``True``, the ``data`` input arument is vector.
    nx : int
        number of x grid locations
    ny : int
        number of y grid locations
    ax : matplotlib.axes
        An axes object
    mask : np.ndarray of bool
        Masking array
    level: bool
        To plot or not
    figname : string
        figure name
    ncontour : int
        number of :meth:`matplotlib.pyplot.contourf` contours
    dataloc : bool
        If ``True``, plot the data locations. If ``False``, do not. Default = ``False``
    contourOpts : dict
        Contour plot options. Dictionary of :meth:`matplotlib.pyplot.contourf` options
    levelOpts : dict
        Level options. Dictionary of :meth:`matplotlib.pyplot.contour` options
    clim : (2) np.ndarray
        Colorbar limits
    method : str {'linear', 'nearest'}
        interpolation method from ``xyz`` locations and the gridded locations for
        the contour plot
    shade : bool
        Add shading to the plot. Default = ``False``
    shade_ncontour : int
        number of :meth:`matplotlib.pyplot.contourf` contours for the shading
    shade_azimuth : float
        azimuth for the light source in shading
    shade_angle_altitude : float
        angle altitude for the light source in shading
    shapeOpts : dict
        Dictionary of :meth:`matplotlib.pyplot.contourf` options

    Returns
    -------
    :meth:`matplotlib.pyplot.contourf`
        Contour object
    matplotlib.axes
        Axes

    """

    # Error checking and set vmin, vmax
    vlimits = [None, None]

    if clim is not None:
        vlimits = [np.min(clim), np.max(clim)]

    for i, key in enumerate(["vmin", "vmax"]):
        if key in contourOpts.keys():
            if vlimits[i] is None:
                vlimits[i] = contourOpts.pop(key)
            else:
                if not np.isclose(contourOpts[key], vlimits[i]):
                    raise Exception(
                        "The values provided in the colorbar limit, clim {} "
                        "does not match the value of {} provided in the "
                        "contourOpts: {}. Only one value should be provided or "
                        "the two values must be equal.".format(
                            vlimits[i], key, contourOpts[key]
                        )
                    )
                contourOpts.pop(key)
    vmin, vmax = vlimits[0], vlimits[1]

    # create a figure if it doesn't exist
    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)

    # interpolate data to grid locations
    xmin, xmax = xyz[:, 0].min(), xyz[:, 0].max()
    ymin, ymax = xyz[:, 1].min(), xyz[:, 1].max()
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)
    xy = np.c_[X.flatten(), Y.flatten()]

    if vec is False:
        if method == "nearest":
            F = NearestNDInterpolator(xyz[:, :2], data)
        else:
            F = LinearNDInterpolator(xyz[:, :2], data)
        DATA = F(xy)
        DATA = DATA.reshape(X.shape)

        # Levels definitions
        dataselection = np.logical_and(~np.isnan(DATA), np.abs(DATA) != np.inf)
        if scale == "log":
            DATA = np.abs(DATA)

        # set vmin, vmax if they are not already set
        vmin = DATA[dataselection].min() if vmin is None else vmin
        vmax = DATA[dataselection].max() if vmax is None else vmax

        if scale == "log":
            levels = np.logspace(np.log10(vmin), np.log10(vmax), ncontour + 1)
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            levels = np.linspace(vmin, vmax, ncontour + 1)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)

        if mask is not None:
            Fmask = NearestNDInterpolator(xyz[:, :2], mask)
            MASK = Fmask(xy)
            MASK = MASK.reshape(X.shape)
            DATA = np.ma.masked_array(DATA, mask=MASK)

        contourOpts = {"levels": levels, "norm": norm, "zorder": 1, **contourOpts}
        cont = ax.contourf(X, Y, DATA, **contourOpts)

        if level:
            levelOpts = {"levels": levels, "zorder": 3, **levelOpts}
            CS = ax.contour(X, Y, DATA, **levelOpts)

    else:
        # Assume size of data is (N,2)
        datax = data[:, 0]
        datay = data[:, 1]
        if method == "nearest":
            Fx = NearestNDInterpolator(xyz[:, :2], datax)
            Fy = NearestNDInterpolator(xyz[:, :2], datay)
        else:
            Fx = LinearNDInterpolator(xyz[:, :2], datax)
            Fy = LinearNDInterpolator(xyz[:, :2], datay)
        DATAx = Fx(xy)
        DATAy = Fy(xy)
        DATA = np.sqrt(DATAx ** 2 + DATAy ** 2).reshape(X.shape)
        DATAx = DATAx.reshape(X.shape)
        DATAy = DATAy.reshape(X.shape)
        if scale == "log":
            DATA = np.abs(DATA)

        # Levels definitions
        dataselection = np.logical_and(~np.isnan(DATA), np.abs(DATA) != np.inf)

        # set vmin, vmax
        vmin = DATA[dataselection].min() if vmin is None else vmin
        vmax = DATA[dataselection].max() if vmax is None else vmax

        if scale == "log":
            levels = np.logspace(np.log10(vmin), np.log10(vmax), ncontour + 1)
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            levels = np.linspace(vmin, vmax, ncontour + 1)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)

        if mask is not None:
            Fmask = NearestNDInterpolator(xyz[:, :2], mask)
            MASK = Fmask(xy)
            MASK = MASK.reshape(X.shape)
            DATA = np.ma.masked_array(DATA, mask=MASK)

        contourOpts = {"levels": levels, "norm": norm, "zorder": 1, **contourOpts}
        cont = ax.contourf(X, Y, DATA, **contourOpts)

        streamplotOpts = {"zorder": 4, "color": "w", **streamplotOpts}
        ax.streamplot(X, Y, DATAx, DATAy, **streamplotOpts)

        if level:
            levelOpts = {"levels": levels, "zorder": 3, **levelOpts}
            CS = ax.contour(X, Y, DATA, levels=levels, zorder=3, **levelOpts)

    if shade:

        def hillshade(array, azimuth, angle_altitude):
            """
            coded copied from https://www.neonscience.org/create-hillshade-py
            """
            azimuth = 360.0 - azimuth
            x, y = np.gradient(array)
            slope = np.pi / 2.0 - np.arctan(np.sqrt(x * x + y * y))
            aspect = np.arctan2(-x, y)
            azimuthrad = azimuth * np.pi / 180.0
            altituderad = angle_altitude * np.pi / 180.0
            shaded = np.sin(altituderad) * np.sin(slope) + np.cos(altituderad) * np.cos(
                slope
            ) * np.cos((azimuthrad - np.pi / 2.0) - aspect)
            return 255 * (shaded + 1) / 2

        shadeOpts = {
            "cmap": "Greys",
            "alpha": 0.35,
            "antialiased": True,
            "zorder": 2,
            **shadeOpts,
        }

        ax.contourf(
            X,
            Y,
            hillshade(DATA, shade_azimuth, shade_angle_altitude),
            shade_ncontour,
            **shadeOpts
        )

    if dataloc:
        ax.plot(xyz[:, 0], xyz[:, 1], "k.", ms=2)
    ax.set_aspect("equal", adjustable="box")
    if figname:
        plt.axis("off")
        fig.savefig(figname, dpi=200)
    if level:
        return cont, ax, CS
    else:
        return cont, ax
