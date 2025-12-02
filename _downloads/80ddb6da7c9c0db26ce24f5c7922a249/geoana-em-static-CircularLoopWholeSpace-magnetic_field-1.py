# Here, we define a horizontal loop and plot the magnetic field
# on the xz-plane that intercepts at y=0.
#
from geoana.em.static import CircularLoopWholeSpace
from geoana.utils import ndgrid
from geoana.plotting_utils import plot2Ddata
import numpy as np
import matplotlib.pyplot as plt
#
# Let us begin by defining the loop.
#
location = np.r_[0., 0., 0.]
orientation = np.r_[0., 0., 1.]
radius = 0.5
simulation = CircularLoopWholeSpace(
    location=location, orientation=orientation, radius=radius
)
#
# Now we create a set of gridded locations and compute the magnetic field.
#
xyz = ndgrid(np.linspace(-1, 1, 50), np.array([0]), np.linspace(-1, 1, 50))
H = simulation.magnetic_field(xyz)
#
# Finally, we plot the magnetic field on the plane.
#
fig = plt.figure(figsize=(4, 4))
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
plot2Ddata(xyz[:, 0::2], H[:, 0::2], ax=ax, vec=True, scale='log')
ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_title('Magnetic field at y=0')
