## Plot the raw_test_matrix.csv file in the directory as a 3D plot
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("raw_test_matrix.csv")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df["design.J"], df["design.alpha"], df["design.delta_e"], color='blue', label='Design Points')

ax.set_xlabel("Angle of Attack")
ax.set_ylabel("Airspeed")
ax.set_zlabel("Elevator Setting")
ax.legend()

plt.show()

