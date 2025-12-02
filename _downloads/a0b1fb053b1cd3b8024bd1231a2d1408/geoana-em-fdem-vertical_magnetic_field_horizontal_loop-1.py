# This example reproduces figure 4.7 from Ward and Hohmann
#
import numpy as np
import matplotlib.pyplot as plt
from geoana.em.fdem import vertical_magnetic_field_horizontal_loop
#
# Define the frequency range,
#
frequencies = np.logspace(-1, 6, 200)
hz = vertical_magnetic_field_horizontal_loop(frequencies, sigma=1E-2, radius=50, secondary=False)
#
# Then plot the values
#
plt.loglog(frequencies, hz.real, c='C0', label='Real')
plt.loglog(frequencies, -hz.real, '--', c='C0')
plt.loglog(frequencies, hz.imag, c='C1', label='Imaginary')
plt.loglog(frequencies, -hz.imag, '--', c='C1')
plt.xlabel('frequency (Hz)')
plt.ylabel('H$_z$ (A/m)')
plt.legend()
plt.show()
