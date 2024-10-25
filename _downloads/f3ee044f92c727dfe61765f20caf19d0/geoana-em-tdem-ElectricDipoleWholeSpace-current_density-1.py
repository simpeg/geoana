# Here, we define an x-oriented electric dipole and plot the current density
# on the xz-plane that intercepts y=0.
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
orientation = np.r_[1., 0., 0.]
current = 1.
sigma = 1.0
simulation = ElectricDipoleWholeSpace(
    time, location=location, orientation=orientation,
    current=current, sigma=sigma
)
#
# Now we create a set of gridded locations and compute the current density.
#
xyz = ndgrid(np.linspace(-10, 10, 20), np.array([0]), np.linspace(-10, 10, 20))
J = simulation.current_density(xyz)
#
# Finally, we plot the current density at the desired locations/times.
#
t_ind = 0
fig = plt.figure(figsize=(4, 4))
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
plot2Ddata(xyz[:, 0::2], J[t_ind, :, 0::2], ax=ax, vec=True, scale='log')
ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_title('Current density at {} s'.format(time[t_ind]))