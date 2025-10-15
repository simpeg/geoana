# Here, we define a point current with current=1A in a wholespace and plot the current density.
#
import numpy as np
import matplotlib.pyplot as plt
from geoana.em.static import PointCurrentWholeSpace
#
# Define the point current.
#
rho = 1.0
current = 1.0
simulation = PointCurrentWholeSpace(
    current=current, rho=rho, location=None
)
#
# Now we create a set of gridded locations and compute the current density.
#
X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
Z = np.zeros_like(X)
xyz = np.stack((X, Y, Z), axis=-1)
j = simulation.current_density(xyz)
#
# Finally, we plot the curent density.
#
j_amp = np.linalg.norm(j, axis=-1)
plt.pcolor(X, Y, j_amp, shading='auto')
cb1 = plt.colorbar()
cb1.set_label(label= 'Current Density ($A/m^2$)')
plt.ylabel('Y coordinate ($m$)')
plt.xlabel('X coordinate ($m$)')
plt.title('Current Density for a Point Current in a Wholespace')
plt.show()
