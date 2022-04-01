import numpy as np
# import properties
from scipy.constants import mu_0

from ..base import BaseLineCurrent


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
