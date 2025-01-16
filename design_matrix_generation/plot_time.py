import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from compute_timings import *

path = "matrix_w_time.csv"
df = pd.read_csv(path)

time = df["Time"]
no_measurements = np.arange(0, len(time)+1, 1)
time = np.append([0], time)

fig, ax = plt.subplots()

ax.plot(time, no_measurements, marker='x', color='r')
ax.axvline(total_tunnel_time, color='k', linestyle='dashed')
ax.set_xlabel('time [s]')
ax.set_ylabel('no. of measurements finished')
ax.grid()
ax.set_xlim(0)
ax.set_ylim(0)
plt.show()