# Here, we define a horizontal loop and plot the vector
# potential on the xy-plane that intercepts at z=0.
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
# Now we create a set of gridded locations and compute the vector potential.
#
xyz = ndgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50), np.array([0]))
a = simulation.vector_potential(xyz)
#
# Finally, we plot the vector potential on the plane. Given the symmetry,
# there are only horizontal components.
#
fig = plt.figure(figsize=(4, 4))
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
plot2Ddata(xyz[:, 0:2], a[:, 0:2], ax=ax, vec=True, scale='log')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Vector potential at z=0')
