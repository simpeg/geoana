"""
Useful reference functions from Ward and Hohmann.

These do not do anything special, and are just direct translations, with
as few modifications as possible, of specific equations from Ward and Hohmann.
"""
from scipy.special import erf, ive, iv
import numpy as np

def hz_from_vert_4_69a(m, theta, rho):
    front = m / (4 * np.pi * rho**3)
    tp = theta * rho
    erf_tp = erf(tp)
    inner_1 = 9 / (2 * tp**2) * erf_tp
    inner_2 = erf_tp
    inner_3 = (9 / tp + 4 * tp)/np.sqrt(np.pi) * np.exp(-tp**2)
    return front * (inner_1 - inner_2 - inner_3)

def dhz_from_vert_4_70(m, theta, rho, sigma, mu):
    front = -m / (2 * np.pi * mu * sigma * rho**5)
    tp = theta * rho
    erf_tp = erf(tp)
    inner_1 = 9 * erf_tp
    inner_2 = 2 * tp / np.sqrt(np.pi) * (
        9 + 6*tp**2 + 4 * tp**4
    ) * np.exp(-tp**2)
    return front * (inner_1 - inner_2)

def hp_from_vert_4_72(m, theta, rho):
    tp = theta * rho
    first = -m * theta**2 / (2 * np.pi * rho)
    second = (ive(1, tp**2/2) - ive(2, tp**2/2))
    return first * second

def dhp_from_vert_4_74(m, theta, rho, t):
    tp = theta * rho
    front = m * theta**2 / (2 * np.pi * rho * t) * np.exp(-tp**2 / 2)

    inner1 = (1 + tp**2)*iv(0, tp**2 / 2)
    inner2 = (2 + tp**2 + 4/tp**2) * iv(1, tp**2/2)

    return front * (inner1 - inner2)