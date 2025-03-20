import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.loadtxt('miscellaneous/DU96_150.dat')
x = data[:, 0]
y = data[:, 1]

# Find the leading edge (minimum x)
ile = np.argmin(x)
# Split into two surfaces
x_lower, y_lower = x[:ile+1], y[:ile+1]
x_upper, y_upper = x[ile:], y[ile:]

# Sort each surface in ascending x order
sort_lower = np.argsort(x_lower)
x_lower, y_lower = x_lower[sort_lower], y_lower[sort_lower]
sort_upper = np.argsort(x_upper)
x_upper, y_upper = x_upper[sort_upper], y_upper[sort_upper]

# Define a common x-grid for interpolation
x_common = np.linspace(np.min(x), np.max(x), 500)
y_lower_interp = np.interp(x_common, x_lower, y_lower)
y_upper_interp = np.interp(x_common, x_upper, y_upper)

# Compute thickness distribution
thickness = y_upper_interp - y_lower_interp
imax = np.argmax(thickness)
x_max = x_common[imax]
print(x_max)
max_thickness = thickness[imax]

# Plot the airfoil and mark the max thickness location
plt.figure(figsize=(10, 5))
plt.plot(x, y, 'b-', label='Airfoil')
plt.plot(x_max, (y_upper_interp[imax] + y_lower_interp[imax]) / 2, 'ro', label='Max Thickness')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Airfoil with Maximum Thickness Location')
plt.legend()
plt.grid(True)
plt.show()

print(f"Maximum thickness: {max_thickness:.4f} at x = {x_max:.4f}")
