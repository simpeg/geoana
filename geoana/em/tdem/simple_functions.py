import numpy as np
from scipy.special import erf, ive
from scipy.constants import mu_0
from geoana.em.tdem.base import theta


def vertical_magnetic_field_horizontal_loop(
    t, sigma=1.0, mu=mu_0, radius=1.0, current=1.0, turns=1
):
    """Vertical transient magnetic field at the center of a horizontal loop over a halfspace.

    Compute the vertical component of the transient magnetic field at the center
    of a circular loop on the surface of a conductive and magnetically permeable halfspace.

    Parameters
    ----------
    t : float, or numpy.ndarray
    sigma : float, optional
        conductivity
    mu : float, optional
        magnetic permeability
    radius : float, optional
        radius of the horizontal loop
    current : float, optional
        current of the horizontal loop
    turns : int, optional
        number of turns in the horizontal loop

    Returns
    -------
    hz : float, or numpy.ndarray
        The vertical magnetic field in H/m at the center of the loop.
        The shape will match the `t` input.

    Notes
    -----
    Equation 4.98 in Ward and Hohmann 1988

    .. math::
        h_z = \\frac{I}{2a}\\left[
            \\frac{3}{\\sqrt{\\pi} \\theta a}e^{-\\theta^2 a^2}
            + \\left(1 - \\frac{3}{2 \\theta^2 a^2}\\right)\\mathrm{erf}(\\theta a)
            \\right]

    Examples
    --------
    Reproducing part of Figure 4.8 from Ward and Hohmann 1988

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from geoana.em.tdem import vertical_magnetic_field_horizontal_loop

    Calculate the field at the time given

    >>> times = np.logspace(-7, -1)
    >>> hz = vertical_magnetic_field_horizontal_loop(times, sigma=1E-2, radius=50)

    Match the vertical magnetic field plot

    >>> plt.loglog(times*1E3, hz)
    >>> plt.xlabel('time (ms)')
    >>> plt.ylabel('H$_z$ (A/m)')
    >>> plt.show()

    """
    theta = np.sqrt((sigma * mu_0) / (4 * t))
    ta = theta * radius
    eta = erf(ta)
    t1 = (3 / (np.sqrt(np.pi) * ta)) * np.exp(-(ta ** 2))
    t2 = (1 - (3 / (2 * ta ** 2))) * eta
    hz = (t1 + t2) / (2 * radius)
    return turns * current * hz


def vertical_magnetic_flux_horizontal_loop(
    t, sigma=1.0, mu=mu_0, radius=1.0, current=1.0, turns=1
):
    """Vertical transient magnetic flux density at the center of a horizontal loop over a halfspace.

    Compute the vertical component of the transient magnetic flux density at the center
    of a circular loop on the surface of a conductive and magnetically permeable halfspace.

    Parameters
    ----------
    t : float, or numpy.ndarray
    sigma : float, optional
        conductivity
    mu : float, optional
        magnetic permeability
    radius : float, optional
        radius of the horizontal loop
    current : float, optional
        current of the horizontal loop
    turns : int, optional
        number of turns in the horizontal loop

    Returns
    -------
    bz : float, or numpy.ndarray
        The vertical magnetic flux density in T/s at the center of the loop.
        The shape will match the `t` input.

    See Also
    --------
    vertical_magnetic_field_horizontal_loop
    """
    return mu * vertical_magnetic_field_horizontal_loop(
        t, sigma=sigma, mu=mu, radius=radius, current=current, turns=turns
    )


def vertical_magnetic_field_time_deriv_horizontal_loop(
    t, sigma=1.0, mu=mu_0, radius=1.0, current=1.0, turns=1
):
    """Time-derivative of the vertical transient magnetic field at the center of a horizontal loop over a halfspace.

    Compute the time-derivative of the vertical component of the transient magnetic field at the center
    of a circular loop on the surface of a conductive and magnetically permeable halfspace.

    Parameters
    ----------
    t : float, or numpy.ndarray
    sigma : float, optional
        conductivity
    mu : float, optional
        magnetic permeability
    radius : float, optional
        radius of the horizontal loop
    current : float, optional
        current of the horizontal loop
    turns : int, optional
        number of turns in the horizontal loop

    Returns
    -------
    dhz_dt : float, or numpy.ndarray
        The vertical magnetic field time derivative at the center of the loop.
        The shape will match the `t` input.

    Notes
    -----
    Matches equation 4.97 of Ward and Hohmann 1988.

    .. math::
        \\frac{\\partial h_z}{\\partial t} = -\\frac{I}{\\sigma a^3}\\left[
        3 \\mathrm{erf}(\\theta a)
        - \\frac{2}{\\sqrt{\\pi}}\\theta a (3 + 2 \\theta^2 a^2)e^{-\\theta^2 a^2}
        \\right]

    Examples
    --------
    Reproducing part of Figure 4.8 from Ward and Hohmann 1988

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from geoana.em.tdem import vertical_magnetic_field_time_deriv_horizontal_loop

    Calculate the field at the time given

    >>> times = np.logspace(-7, -1)
    >>> dhz_dt = vertical_magnetic_field_time_deriv_horizontal_loop(times, sigma=1E-2, radius=50)

    Match the vertical magnetic field plot

    >>> plt.loglog(times*1E3, -dhz_dt, '--')
    >>> plt.xlabel('time (ms)')
    >>> plt.ylabel(r'$\\frac{\\partial h_z}{ \\partial t}$ (A/(m s)')
    >>> plt.show()
    """
    a = radius
    the = theta(t, sigma, mu)
    return -turns * current / (mu * sigma * a**3) * (
        3*erf(the * a) - 2/np.sqrt(np.pi) * the * a * (3 + 2 * the**2 * a**2) * np.exp(-the**2 * a**2)
    )


def vertical_magnetic_flux_time_deriv_horizontal_loop(
    t, sigma=1.0, mu=mu_0, radius=1.0, current=1.0, turns=1
):
    """Time-derivative of the vertical transient magnetic flux density at the center of a horizontal loop over a halfspace.

    Compute the time-derivative of the vertical component of the transient
    magnetic flux density at the center of a circular loop on the surface
    of a conductive and magnetically permeable halfspace.

    Parameters
    ----------
    t : float, or numpy.ndarray
    sigma : float, optional
        conductivity
    mu : float, optional
        magnetic permeability
    radius : float, optional
        radius of the horizontal loop
    current : float, optional
        current of the horizontal loop
    turns : int, optional
        number of turns in the horizontal loop

    Returns
    -------
    dbz_dt : float, or numpy.ndarray
        The vertical magnetic flux time derivative at the center of the loop.
        The shape will match the `t` input.

    See Also
    --------
    vertical_magnetic_field_time_deriv_horizontal_loop

    Examples
    --------
    Reproducing part of Figure 4.8, scaled by magnetic suscpetibility, from Ward and
    Hohmann 1988.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from geoana.em.tdem import vertical_magnetic_flux_time_deriv_horizontal_loop

    Calculate the field at the time given

    >>> times = np.logspace(-7, -1)
    >>> dbz_dt = vertical_magnetic_flux_time_deriv_horizontal_loop(times, sigma=1E-2, radius=50)

    Match the vertical magnetic field plot

    >>> plt.loglog(times*1E3, -dbz_dt, '--')
    >>> plt.xlabel('time (ms)')
    >>> plt.ylabel(r'$\\frac{\\partial b_z}{ \\partial t}$ (T/s)')
    >>> plt.show()
    """
    a = radius
    the = theta(t, sigma, mu)
    return -turns * current / (sigma * a**3) * (
        3*erf(the * a) - 2/np.sqrt(np.pi) * the * a * (3 + 2 * the**2 * a**2) * np.exp(-the**2 * a**2)
    )


def magnetic_field_vertical_magnetic_dipole(
    t, xy, sigma=1.0, mu=mu_0, moment=1.0
):
    """Magnetic field due to step off vertical dipole at the surface

    Parameters
    ----------
    t : (n_t) numpy.ndarray
        times (s)
    xy : (n_locs, 2) numpy.ndarray
        surface field locations (m)
    sigma : float, optional
        conductivity
    mu : float, optional
        magnetic permeability
    moment : float, optional
        moment of the dipole

    Returns
    -------
    h : (n_t, n_locs, 3) numpy.ndarray
        The magnetic field at the observation locations and times.

    Notes
    -----
    Matches the negative of equation 4.69a of Ward and Hohmann 1988, for the vertical
    component (due to the difference in coordinate sign conventionn used here).

    .. math::
        h_z = -\\frac{m}{4 \\pi \\rho^2} \\left[
        \\left(\\frac{9}{2 \\theta^2 \\rho^2} - 1\\right)\\mathrm{erf}(\\theta \\rho)
        - \\frac{1}{\\sqrt{\\pi}}\\left(\\frac{9}{\\theta \\rho + 4 \\theta \\rho} \\right)
        e^{-\\theta^2\\rho^2}
        \\right]

    Also matches equation 4.72 for the horizontal components, which is again negative due
    to our coordinate convention.

    .. math::
        h_\\rho = \\frac{m \\theta^2}{2\\pi\\rho}
        e^{-\\theta^2\\rho^2/2}\\left[I_1\\left(\\frac{\\theta^2\\rho^2}{2}\\right)
         - I_2\\left(\\frac{\\theta^2\\rho^2}{2}\\right)\\right]

    Examples
    --------
    Reproducing part of Figure 4.4 and 4.5 from Ward and Hohmann 1988

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from geoana.em.tdem import magnetic_field_vertical_magnetic_dipole

    Calculate the field at the time given, and 100 m along the x-axis,

    >>> times = np.logspace(-8, 0, 200)
    >>> xy = np.array([[100, 0, 0]])
    >>> h = magnetic_field_vertical_magnetic_dipole(times, xy, sigma=1E-2)

    Match the vertical magnetic field plot

    >>> plt.loglog(times*1E3, h[:,0, 2], c='C0', label='$h_z$')
    >>> plt.loglog(times*1E3, -h[:,0, 2], '--', c='C0')
    >>> plt.loglog(times*1E3, h[:,0, 0], c='C1', label='$h_x$')
    >>> plt.loglog(times*1E3, -h[:,0, 0], '--', c='C1')
    >>> plt.xlabel('time (ms)')
    >>> plt.ylabel('h (A/m)')
    >>> plt.legend()
    >>> plt.show()
    """
    r = np.linalg.norm(xy[:, :2], axis=-1)
    x = xy[:, 0]
    y = xy[:, 1]
    thr = theta(t, sigma, mu=mu)[:, None] * r #will be positive...

    h_z = 1.0 / r**3 * (
        (9 / (2 * thr**2) - 1) * erf(thr)
        - (9 / thr + 4 * thr) / np.sqrt(np.pi) * np.exp(-thr**2)
    )
    # positive here because z+ up

    # iv(1, arg) - iv(2, arg)
    # ive(1, arg) * np.exp(abs(arg)) - ive(2, arg) * np.exp(abs(arg))
    # (ive(1, arg) - ive(2, arg))*np.exp(abs(arg))
    h_r = 2 * thr**2 / r**3 * (
        ive(1, thr**2 / 2) - ive(2, thr**2 / 2)
    )
    # thetar is always positive so this above simplifies (more numerically stable)

    angle = np.arctan2(y, x)
    h_x = np.cos(angle) * h_r
    h_y = np.sin(angle) * h_r

    return moment / (4 * np.pi) * np.stack((h_x, h_y, h_z), axis=-1)


def magnetic_field_time_deriv_magnetic_dipole(
    t, xy, sigma=1.0, mu=mu_0, moment=1.0
):
    """Magnetic field time derivative due to step off vertical dipole at the surface

    Parameters
    ----------
    t : (n_t) numpy.ndarray
        times (s)
    xy : (n_locs, 2) numpy.ndarray
        surface field locations (m)
    sigma : float, optional
        conductivity
    mu : float, optional
        magnetic permeability
    moment : float, optional
        moment of the dipole

    Returns
    -------
    dh_dt : (n_t, n_locs, 3) numpy.ndarray
        The magnetic field at the observation locations and times.

    Notes
    -----
    Matches the negative of equation 4.70 of Ward and Hohmann 1988, for the vertical
    component (due to the difference in coordinate sign conventionn used here).

    .. math::
        \\frac{\\partial h_z}{\\partial t} = \\frac{m}{2 \\pi \\mu \\sigma \\rho^5}\\left[
        9\\mathrm{erf}(\\theta \\rho)
        - \\frac{2\\theta\\rho}{\\sqrt{\\pi}}\\left(
        9 + 6\\theta^2\\rho^2 + 4\\theta^4\\rho^4\\right)
        e^{-\\theta^2\\rho^2}
        \\right]

    Also matches equation 4.74 for the horizontal components

    .. math::
        \\frac{\\partial h_\\rho}{\\partial t} = -\\frac{m \\theta^2}{2 \\pi \\rho t}e^{-\\theta^2\\rho^2/2} \\left[
        (1 +\\theta^2 \\rho^2) I_0 \\left(
        \\frac{\\theta^2\\rho^2}{2}
        \\right) - \\left(
        2 + \\theta^2\\rho^2 + \\frac{4}{\\theta^2\\rho^2}\\right)I_1\\left(\\frac{\\theta^2\\rho^2}{2}\\right)
        \\right]

    Examples
    --------
    Reproducing the time derivate parts of Figure 4.4 and 4.5 from Ward and Hohmann 1988

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from geoana.em.tdem import magnetic_field_time_deriv_magnetic_dipole

    Calculate the field at the time given, 100 m along the x-axis,

    >>> times = np.logspace(-6, 0, 200)
    >>> xy = np.array([[100, 0, 0]])
    >>> dh_dt = magnetic_field_time_deriv_magnetic_dipole(times, xy, sigma=1E-2)

    Match the vertical magnetic field plot

    >>> plt.loglog(times*1E3, dh_dt[:,0, 2], c='C0', label=r'$\\frac{\\partial h_z}{\\partial t}$')
    >>> plt.loglog(times*1E3, -dh_dt[:,0, 2], '--', c='C0')
    >>> plt.loglog(times*1E3, dh_dt[:,0, 0], c='C1', label=r'$\\frac{\\partial h_x}{\\partial t}$')
    >>> plt.loglog(times*1E3, -dh_dt[:,0, 0], '--', c='C1')
    >>> plt.xlabel('time (ms)')
    >>> plt.ylabel(r'$\\frac{\\partial h}{\\partial t}$ (A/(m s))')
    >>> plt.legend()
    >>> plt.show()
    """
    r = np.linalg.norm(xy[:, :2], axis=-1)
    x = xy[:, 0]
    y = xy[:, 1]
    tr = theta(t, sigma, mu)[:, None] * r

    dhz_dt = 1 / (r**3 * t[:, None]) * (
        9 / (2 * tr**2) * erf(tr)
        - (4 * tr**3 + 6 * tr + 9/tr)/np.sqrt(np.pi)*np.exp(-tr**2)
    )

    # iv(k, v) = ive(k, v) * exp(abs(arg))
    dhr_dt = - 2 * tr**2 / (r**3 * t[:, None]) * (
        (1 + tr**2) * ive(0, tr**2 / 2) -
        (2 + tr**2 + 4 / tr**2) * ive(1, tr**2 / 2)
    )
    angle = np.arctan2(y, x)
    dhx_dt = np.cos(angle) * dhr_dt
    dhy_dt = np.sin(angle) * dhr_dt
    return moment / (4 * np.pi) * np.stack((dhx_dt, dhy_dt, dhz_dt), axis=-1)


def magnetic_flux_vertical_magnetic_dipole(
    t, xy, sigma=1.0, mu=mu_0, moment=1.0
):
    """
    Magnetic flux density due to step off vertical dipole at the surface

    Parameters
    ----------
    t : (n_t) numpy.ndarray
        times (s)
    xy : (n_locs, 2) numpy.ndarray
        surface field locations (m)
    sigma : float, optional
        conductivity
    mu : float, optional
        magnetic permeability
    moment : float, optional
        moment of the dipole

    Returns
    -------
    b : (n_t, n_locs, 3) numpy.ndarray
        The magnetic flux at the observation locations and times.

    See Also
    --------
    magnetic_field_vertical_magnetic_dipole
    """
    return mu * magnetic_field_vertical_magnetic_dipole(
        t, xy, sigma=sigma, mu=mu, moment=moment
    )


def magnetic_flux_time_deriv_magnetic_dipole(
    t, xy, sigma=1.0, mu=mu_0, moment=1.0
):
    """
    Magnetic flux density time derivative due to step off vertical dipole at the surface

    Parameters
    ----------
    t : (n_t) numpy.ndarray
        times (s)
    xy : (n_locs, 2) numpy.ndarray
        surface field locations (m)
    sigma : float, optional
        conductivity
    mu : float, optional
        magnetic permeability
    moment : float, optional
        moment of the dipole

    Returns
    -------
    db_dt : (n_t, n_locs, 3) numpy.ndarray
        The magnetic flux at the observation locations and times.

    See Also
    --------
    magnetic_field_time_deriv_magnetic_dipole
    """
    return mu * magnetic_field_time_deriv_magnetic_dipole(
        t, xy, sigma=sigma, mu=mu, moment=moment
    )
