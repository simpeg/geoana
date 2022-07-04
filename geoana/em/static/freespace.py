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


class LineCurrentFreeSpace(BaseLineCurrent):
    """Class for a static line current in free space.

    The ``LineCurrentFreeSpace`` class is used to analytically compute the
    fields and potentials within freespace due to a set of static current-carrying wires.
    """

    def magnetic_field(self, xyz):
        r"""Compute the magnetic field for the static current-carrying wire segments.

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray xyz
            gridded locations at which we are calculating the magnetic field

        Returns
        -------
        (n, 3) numpy.ndarray
            The magnetic field at each observation location in H/m.

        Examples
        --------
        Here, we define a horizontal square loop and plot the magnetic field
        on the xz-plane that intercepts at y=0.

        >>> from geoana.em.static import LineCurrentFreeSpace
        >>> from geoana.utils import ndgrid
        >>> from geoana.plotting_utils import plot2Ddata
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the loop. Note that to create an inductive
        source, we closed the loop.

        >>> x_nodes = np.array([-0.5, 0.5, 0.5, -0.5, -0.5])
        >>> y_nodes = np.array([-0.5, -0.5, 0.5, 0.5, -0.5])
        >>> z_nodes = np.zeros_like(x_nodes)
        >>> nodes = np.c_[x_nodes, y_nodes, z_nodes]
        >>> simulation = LineCurrentFreeSpace(nodes)

        Now we create a set of gridded locations and compute the magnetic field.

        >>> xyz = ndgrid(np.linspace(-1, 1, 50), np.array([0]), np.linspace(-1, 1, 50))
        >>> H = simulation.magnetic_field(xyz)

        Finally, we plot the magnetic field.

        >>> fig = plt.figure(figsize=(4, 4))
        >>> ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])
        >>> plot2Ddata(xyz[:, [0, 2]], H[:, [0, 2]], ax=ax, vec=True, scale='log', ncontour=25)
        >>> ax.set_xlabel('X')
        >>> ax.set_ylabel('Z')
        >>> ax.set_title('Magnetic field')

        """

        # TRANSMITTER NODES
        I = self.current
        tx_nodes = self.nodes
        x1tr = tx_nodes[:, 0]
        x2tr = tx_nodes[:, 1]
        x3tr = tx_nodes[:, 2]

        n_loc = np.shape(xyz)[0]
        n_segments = self.n_segments

        hx0 = np.zeros(n_loc)
        hy0 = np.zeros(n_loc)
        hz0 = np.zeros(n_loc)

        for pp in range(0, n_segments):

            # Wire ends for transmitter wire pp
            x1a = x1tr[pp]
            x2a = x2tr[pp]
            x3a = x3tr[pp]
            x1b = x1tr[pp + 1]
            x2b = x2tr[pp + 1]
            x3b = x3tr[pp + 1]

            # Vector Lengths between points
            vab = np.sqrt((x1b - x1a) ** 2 + (x2b - x2a) ** 2 + (x3b - x3a) ** 2)
            vap = np.sqrt(
                (xyz[:, 0] - x1a) ** 2 + (xyz[:, 1] - x2a) ** 2 + (xyz[:, 2] - x3a) ** 2
            )
            vbp = np.sqrt(
                (xyz[:, 0] - x1b) ** 2 + (xyz[:, 1] - x2b) ** 2 + (xyz[:, 2] - x3b) ** 2
            )

            # Cosines from cos()=<v1,v2>/(|v1||v2|)
            cos_alpha = (
                (xyz[:, 0] - x1a) * (x1b - x1a)
                + (xyz[:, 1] - x2a) * (x2b - x2a)
                + (xyz[:, 2] - x3a) * (x3b - x3a)
            ) / (vap * vab)
            cos_beta = (
                (xyz[:, 0] - x1b) * (x1a - x1b)
                + (xyz[:, 1] - x2b) * (x2a - x2b)
                + (xyz[:, 2] - x3b) * (x3a - x3b)
            ) / (vbp * vab)

            # Determining Radial Vector From Wire
            dot_temp = (
                (x1a - xyz[:, 0]) * (x1b - x1a)
                + (x2a - xyz[:, 1]) * (x2b - x2a)
                + (x3a - xyz[:, 2]) * (x3b - x3a)
            )

            rx1 = (x1a - xyz[:, 0]) - dot_temp * (x1b - x1a) / vab ** 2
            rx2 = (x2a - xyz[:, 1]) - dot_temp * (x2b - x2a) / vab ** 2
            rx3 = (x3a - xyz[:, 2]) - dot_temp * (x3b - x3a) / vab ** 2

            r = np.sqrt(rx1 ** 2 + rx2 ** 2 + rx3 ** 2)

            phi = (cos_alpha + cos_beta) / r

            # I/4*pi in each direction
            ix1 = I * (x1b - x1a) / (4 * np.pi * vab)
            ix2 = I * (x2b - x2a) / (4 * np.pi * vab)
            ix3 = I * (x3b - x3a) / (4 * np.pi * vab)

            # Add contribution from wire pp into array
            hx0 = hx0 + phi * (-ix2 * rx3 + ix3 * rx2) / r
            hy0 = hy0 + phi * (ix1 * rx3 - ix3 * rx1) / r
            hz0 = hz0 + phi * (-ix1 * rx2 + ix2 * rx1) / r

        return np.c_[hx0, hy0, hz0]

    def magnetic_flux_density(self, xyz):
        r"""Compute the magnetic flux density for the static current-carrying wire segments.

        Parameters
        ----------
        xyz : (n, 3) numpy.ndarray xyz
            gridded locations at which we are calculating the magnetic flux density

        Returns
        -------
        (n, 3) numpy.ndarray
            The magnetic flux density at each observation location in T.

        Examples
        --------
        Here, we define a horizontal square loop and plot the magnetic flux
        density on the XZ-plane that intercepts at Y=0.

        >>> from geoana.em.static import LineCurrentFreeSpace
        >>> from geoana.utils import ndgrid
        >>> from geoana.plotting_utils import plot2Ddata
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        Let us begin by defining the loop. Note that to create an inductive
        source, we closed the loop

        >>> x_nodes = np.array([-0.5, 0.5, 0.5, -0.5, -0.5])
        >>> y_nodes = np.array([-0.5, -0.5, 0.5, 0.5, -0.5])
        >>> z_nodes = np.zeros_like(x_nodes)
        >>> nodes = np.c_[x_nodes, y_nodes, z_nodes]
        >>> simulation = LineCurrentFreeSpace(nodes)

        Now we create a set of gridded locations and compute the magnetic flux density.

        >>> xyz = ndgrid(np.linspace(-1, 1, 50), np.array([0]), np.linspace(-1, 1, 50))
        >>> B = simulation.magnetic_flux_density(xyz)

        Finally, we plot the magnetic flux density on the plane.

        >>> fig = plt.figure(figsize=(4, 4))
        >>> ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
        >>> plot2Ddata(xyz[:, [0, 2]], B[:, [0, 2]], ax=ax, vec=True, scale='log', ncontour=25)
        >>> ax.set_xlabel('X')
        >>> ax.set_ylabel('Z')
        >>> ax.set_title('Magnetic flux density')

        """

        return mu_0 * self.magnetic_field(xyz)


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
        return mu_0 * self.magnetic_field(xyz)

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
