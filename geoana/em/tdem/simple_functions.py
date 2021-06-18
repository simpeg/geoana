import numpy as np
from scipy.special import erf, iv, erfc
from scipy.constants import mu_0
from geoana.em.tdem.base import theta


def vertical_magnetic_field_horizontal_loop(
    t, sigma=1.0, mu=mu_0, radius=1.0, current=1.0, turns=1
):
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
    return mu * vertical_magnetic_field_horizontal_loop(
        t, sigma=sigma, mu=mu, radius=radius, current=current, turns=turns
    )


def vertical_magnetic_flux_time_deriv_horizontal_loop(
    t, sigma=1.0, mu=mu_0, radius=1.0, current=1.0, turns=1
):
    a = radius
    the = theta(t, sigma, mu)
    return -turns * current / (sigma * a**3) * (
        3*erf(the * a) - 2/np.sqrt(np.pi) * the * a * (3 + 2 * the**2 * a**2) * np.exp(-the**2 * a**2)
    )


def magnetic_field_vertical_magnetic_dipole(
    t, xy, sigma=1.0, mu=mu_0, moment=1.0
):
    r = np.linalg.norm(xy[:, :2], axis=-1)
    x = xy[:, 0]
    y = xy[:, 1]
    thr = theta(t, sigma, mu=mu)[:, None] * r

    h_z = 1.0 / r**3 * (
        (9 / (2 * thr**2) - 1) * erf(thr)
        - (9 / thr + 4 * thr) / np.sqrt(np.pi) * np.exp(-thr**2)
    )
    # positive here because z+ up
    h_r = 2 * thr**2 / r**3 * np.exp(-thr**2 / 2) * (
        iv(1, thr**2 / 2) - iv(2, thr**2 / 2)
    )
    angle = np.arctan2(y, x)
    h_x = np.cos(angle) * h_r
    h_y = np.sin(angle) * h_r

    return moment / (4 * np.pi) * np.stack((h_x, h_y, h_z), axis=-1)


def magnetic_field_time_deriv_magnetic_dipole(
    t, xy, sigma=1.0, mu=mu_0, moment=1.0
):
    r = np.linalg.norm(xy[:, :2], axis=-1)
    x = xy[:, 0]
    y = xy[:, 1]
    tr = theta(t, sigma, mu)[:, None] * r

    dhz_dt = 1 / (r**3 * t[:, None]) * (
        9 / (2 * tr**2) * erf(tr)
        - (4 * tr**3 + 6 * tr + 9/tr)/np.sqrt(np.pi)*np.exp(-tr**2)
    )

    dhr_dt = - 2 * tr**2 / (r**3 * t[:, None]) * np.exp(-tr**2 / 2) * (
        (1 + tr**2) * iv(0, tr**2 / 2) -
        (2 + tr**2 + 4 / tr**2) * iv(1, tr**2 / 2)
    )
    angle = np.arctan2(y, x)
    dhx_dt = np.cos(angle) * dhr_dt
    dhy_dt = np.sin(angle) * dhr_dt
    return moment / (4 * np.pi) * np.stack((dhx_dt, dhy_dt, dhz_dt), axis=-1)


def magnetic_flux_vertical_magnetic_dipole(
    t, xy, sigma=1.0, mu=mu_0, moment=1.0
):
    return mu * magnetic_field_vertical_magnetic_dipole(
        t, xy, sigma=sigma, mu=mu, moment=moment
    )


def magnetic_flux_time_deriv_magnetic_dipole(
    t, xy, sigma=1.0, mu=mu_0, moment=1.0
):
    return mu * magnetic_field_time_deriv_magnetic_dipole(
        t, xy, sigma=sigma, mu=mu, moment=moment
    )
