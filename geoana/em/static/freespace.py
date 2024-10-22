import numpy as np
from scipy.constants import mu_0

from ..base import BaseLineCurrent
from geoana.utils import check_xyz_dim
from geoana.shapes import BasePrism
from geoana.kernels import (
    prism_fz,
    prism_fzz,
    prism_fzx,
    prism_fzy,
    prism_fzzz,
    prism_fxxy,
    prism_fxxz,
    prism_fxyz,
)


class MagneticPrism(BasePrism):
    """Class for magnetic field solutions for a prism.

    The ``Prism`` class is used to analytically compute the magnetic
    potentials, fields, and gradients for a prism with constant magnetization.

    Parameters
    ----------
    min_location : (3,) array_like
        (x, y, z) triplet of the minimum locations in each dimension
    max_location : (3,) array_like
        (x, y, z) triplet of the maximum locations in each dimension
    magnetization : (3,) array_like, optional
        Magnetization of prism (:math:`\\frac{A}{m}`).
    """

    def __init__(self, min_location, max_location, magnetization=None):

        if magnetization is None:
            magnetization = np.r_[0.0, 0.0, 1.0]
        self.magnetization = magnetization

        super().__init__(min_location=min_location, max_location=max_location)

    @property
    def magnetization(self):
        return self._magnetization

    @magnetization.setter
    def magnetization(self, vec):
        try:
            vec = np.asarray(vec, dtype=float)
        except:
            raise TypeError(f"location must be array_like of float, got {type(vec)}")

        vec = np.squeeze(vec)
        if vec.shape != (3,):
            raise ValueError(
                f"magnetization must be array_like with shape (3,), got {vec.shape}"
            )

        self._magnetization = vec

    @property
    def moment(self):
        return self.volume * self.magnetization

    def scalar_potential(self, xyz):
        """
        Magnetic scalar potential due to a prism. Defined such that
        :math:`H = \\nabla \\phi`.

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Observation locations in units m.

        Returns
        -------
        (...) numpy.ndarray
            Magnetic scalar potential of prism at location xyz in units :math:`A`.
        """

        xyz = check_xyz_dim(xyz)
        m_x, m_y, m_z = self.magnetization

        gx = self._eval_def_int(prism_fz, xyz[..., 0], xyz[..., 1], xyz[..., 2], cycle=1)
        gy = self._eval_def_int(prism_fz, xyz[..., 0], xyz[..., 1], xyz[..., 2], cycle=2)
        gz = self._eval_def_int(prism_fz, xyz[..., 0], xyz[..., 1], xyz[..., 2])

        return -1.0/(4 * np.pi) * (gx * m_x + gy * m_y + gz * m_z)

    def magnetic_field(self, xyz):
        """
        Magnetic field due to a prism.

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Point mass location in units m.

        Returns
        -------
        (..., 3) numpy.ndarray
            Magnetic field at point mass location xyz in units :math:`\\frac{A}{m}`.
        """

        xyz = check_xyz_dim(xyz)
        m_x, m_y, m_z = self.magnetization

        # need to evaluate f node at each source locations
        gxx = self._eval_def_int(prism_fzz, xyz[..., 0], xyz[..., 1], xyz[..., 2], cycle=1)
        gxy = self._eval_def_int(prism_fzx, xyz[..., 0], xyz[..., 1], xyz[..., 2], cycle=1)
        gxz = self._eval_def_int(prism_fzx, xyz[..., 0], xyz[..., 1], xyz[..., 2])

        gyy = self._eval_def_int(prism_fzz, xyz[..., 0], xyz[..., 1], xyz[..., 2], cycle=2)
        gyz = self._eval_def_int(prism_fzy, xyz[..., 0], xyz[..., 1], xyz[..., 2])

        # gzz = - gxx - gyy - 4 * np.pi * G * rho[in_cell]
        # easiest to just calculate it using another integral
        gzz = self._eval_def_int(prism_fzz, xyz[..., 0], xyz[..., 1], xyz[..., 2])

        H = - 1.0/(4 * np.pi) * np.stack(
            (
                gxx * m_x + gxy * m_y + gxz * m_z,
                gxy * m_x + gyy * m_y + gyz * m_z,
                gxz * m_x + gyz * m_y + gzz * m_z
            ),
            axis=-1
        )
        return H

    def magnetic_flux_density(self, xyz):
        """
        Magnetic field due to a prism.

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Point mass location in units m.

        Returns
        -------
        (..., 3) numpy.ndarray
            Magnetic flux density or prism at location xyz in units :math:`T`.
        """
        xyz = check_xyz_dim(xyz)
        H = self.magnetic_field(xyz)
        is_inside = (
            np.all(xyz >= self.min_location, axis=-1)
            & np.all(xyz <= self.max_location, axis=-1)
        )
        H[is_inside] = H[is_inside] + self.magnetization

        return mu_0 * H

    def magnetic_field_gradient(self, xyz):
        """
        Magnetic field gradient due to a prism.

        Parameters
        ----------
        xyz : (..., 3) numpy.ndarray
            Observation locations in units m.

        Returns
        -------
        (..., 3, 3) numpy.ndarray
            Magnetic field gradient of prism at location xyz in units :math:`\\frac{A}{m^2}`.
        """

        xyz = check_xyz_dim(xyz)
        m_x, m_y, m_z = self.magnetization

        # need to evaluate f node at each source locations
        gxxx = self._eval_def_int(prism_fzzz, xyz[..., 0], xyz[..., 1], xyz[..., 2], cycle=1)
        gxxy = self._eval_def_int(prism_fxxy, xyz[..., 0], xyz[..., 1], xyz[..., 2])
        gxxz = self._eval_def_int(prism_fxxz, xyz[..., 0], xyz[..., 1], xyz[..., 2])
        gyyx = self._eval_def_int(prism_fxxz, xyz[..., 0], xyz[..., 1], xyz[..., 2], cycle=1)
        gxyz = self._eval_def_int(prism_fxyz, xyz[..., 0], xyz[..., 1], xyz[..., 2])
        gzzx = self._eval_def_int(prism_fxxy, xyz[..., 0], xyz[..., 1], xyz[..., 2], cycle=2)
        gyyy = self._eval_def_int(prism_fzzz, xyz[..., 0], xyz[..., 1], xyz[..., 2], cycle=2)
        gyyz = self._eval_def_int(prism_fxxy, xyz[..., 0], xyz[..., 1], xyz[..., 2], cycle=1)
        gzzy = self._eval_def_int(prism_fxxz, xyz[..., 0], xyz[..., 1], xyz[..., 2], cycle=2)
        gzzz = self._eval_def_int(prism_fzzz, xyz[..., 0], xyz[..., 1], xyz[..., 2])

        Hxx = gxxx * m_x + gxxy * m_y + gxxz * m_z
        Hxy = gxxy * m_x + gyyx * m_y + gxyz * m_z
        Hxz = gxxz * m_x + gxyz * m_y + gzzx * m_z
        Hyy = gyyx * m_x + gyyy * m_y + gyyz * m_z
        Hyz = gxyz * m_x + gyyz * m_y + gzzy * m_z
        Hzz = gzzx * m_x + gzzy * m_y + gzzz * m_z

        first = np.stack([Hxx, Hxy, Hxz], axis=-1)
        second = np.stack([Hxy, Hyy, Hyz], axis=-1)
        third = np.stack([Hxz, Hyz, Hzz], axis=-1)

        H_grad = - 1.0/(4 * np.pi) * np.stack((first, second, third), axis=-1)
        return H_grad
