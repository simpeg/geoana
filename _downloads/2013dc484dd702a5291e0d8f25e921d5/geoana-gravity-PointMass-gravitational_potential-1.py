# Here, we define a point mass with mass=1kg and plot the gravitational
# potential as a function of distance.
#
import numpy as np
import matplotlib.pyplot as plt
from geoana.gravity import PointMass
#
# Define the point mass.
#
location = np.r_[0., 0., 0.]
mass = 1.0
simulation = PointMass(
    mass=mass, location=location
)
#
# Now we create a set of gridded locations, take the distances and compute the gravitational potential.
#
X, Y = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
Z = np.zeros_like(X) + 0.25
xyz = np.stack((X, Y, Z), axis=-1)
r = np.linalg.norm(xyz, axis=-1)
u = simulation.gravitational_potential(xyz)
#
# Finally, we plot the gravitational potential as a function of distance.
#
plt.plot(r, u)
plt.xlabel('Distance from point mass')
plt.ylabel('Gravitational potential')
plt.title('Gravitational Potential as a function of distance from point mass')
plt.show()
