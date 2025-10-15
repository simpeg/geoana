# Here, we define an z-oriented magnetic dipole and plot the magnetic field
# on the xz-plane that intercepts y=0.
#
from geoana.em.fdem import MagneticDipoleWholeSpace
from geoana.utils import ndgrid
from geoana.plotting_utils import plot2Ddata
import numpy as np
import matplotlib.pyplot as plt
#
# Let us begin by defining the electric current dipole.
#
frequency = np.logspace(1, 3, 3)
location = np.r_[0., 0., 0.]
orientation = np.r_[0., 0., 1.]
moment = 1.
sigma = 1.0
simulation = MagneticDipoleWholeSpace(
    frequency, location=location, orientation=orientation,
    moment=moment, sigma=sigma
)
#
# Now we create a set of gridded locations and compute the magnetic field.
#
xyz = ndgrid(np.linspace(-1, 1, 20), np.array([0]), np.linspace(-1, 1, 20))
H = simulation.magnetic_field(xyz)
#
# Finally, we plot the real and imaginary components of the magnetic field.
#
f_ind = 2
fig = plt.figure(figsize=(6, 3))
ax1 = fig.add_axes([0.15, 0.15, 0.40, 0.75])
plot2Ddata(
    xyz[:, 0::2], np.real(H[f_ind, :, 0::2]), vec=True, ax=ax1, scale='log', ncontour=25
)
ax1.set_xlabel('X')
ax1.set_ylabel('Z')
ax1.autoscale(tight=True)
ax1.set_title('Real component {} Hz'.format(frequency[f_ind]))
ax2 = fig.add_axes([0.6, 0.15, 0.40, 0.75])
plot2Ddata(
    xyz[:, 0::2], np.imag(H[f_ind, :, 0::2]), vec=True, ax=ax2, scale='log', ncontour=25
)
ax2.set_xlabel('X')
ax2.set_yticks([])
ax2.autoscale(tight=True)
ax2.set_title('Imag component {} Hz'.format(frequency[f_ind]))
