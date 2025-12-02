# Reproducing part of Figure 4.8 from Ward and Hohmann 1988
#
import numpy as np
import matplotlib.pyplot as plt
from geoana.em.tdem import vertical_magnetic_field_horizontal_loop
#
# Calculate the field at the time given
#
times = np.logspace(-7, -1)
hz = vertical_magnetic_field_horizontal_loop(times, sigma=1E-2, radius=50)
#
# Match the vertical magnetic field plot
#
plt.loglog(times*1E3, hz)
plt.xlabel('time (ms)')
plt.ylabel('H$_z$ (A/m)')
plt.show()
