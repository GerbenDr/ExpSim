
import numpy as np

baseline_tunnel = 102.5
J_variation = 2
# reflections = -20
reflections = 0
Mach_effect = 38.5 * (0.73 - 0.3)


scaling = 18.3

total = baseline_tunnel + J_variation + reflections + Mach_effect + scaling 
print(total)

