# Here, we define a z-oriented electric dipole and plot the magnetic field
# on the xy-plane that intercepts z=0.
#
from geoana.em.tdem import ElectricDipoleWholeSpace
from geoana.utils import ndgrid
from geoana.plotting_utils import plot2Ddata
import numpy as np
import matplotlib.pyplot as plt
#
# Let us begin by defining the electric current dipole.
#
time = np.logspace(-6, -2, 3)
location = np.r_[0., 0., 0.]
orientation = np.r_[0., 0., 1.]
current = 1.
sigma = 1.0
simulation = ElectricDipoleWholeSpace(
    time, location=location, orientation=orientation,
    current=current, sigma=sigma
)
#
# Now we create a set of gridded locations and compute the magnetic field.
#
xyz = ndgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20), np.array([0]))
H = simulation.magnetic_field(xyz)
#
# Finally, we plot the magnetic field at the desired locations/times.
#
t_ind = 0
fig = plt.figure(figsize=(4, 4))
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
plot2Ddata(xyz[:, 0:2], H[t_ind, :, 0:2], ax=ax, vec=True, scale='log')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Magnetic field at {} s'.format(time[t_ind]))
