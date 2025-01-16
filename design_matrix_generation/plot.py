
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

df = pd.read_csv('design_matrix_generation/raw_test_matrix.csv')


# Plot the boxbhenken design points in red. Plot the optimized values in blue.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot optimized points in blue
ax.scatter(df["design.alpha"], df["design.J"], df["design.delta_e"], color='blue', label='Optimized Points')

ax.set_xlabel("Angle of Attack")
ax.set_ylabel("Advance Ratio")
ax.set_zlabel("Flap Setting")
ax.legend()

plt.show()