import numpy as np
from scipy.interpolate import interp1d

# Extracted data from the file
# based on xfoil using:
# xtr = 0.05 for top and bottom
# Ncrit = 9
# Re = 419512
alpha_data = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
    5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.5, 9.0, 9.5, 10.0
])

cl_data = np.array([
   -0.0000, 0.0535, 0.1070, 0.1604, 0.2137, 0.2669, 0.3198, 0.3726, 0.4250, 0.4770,
    0.5286, 0.5796, 0.6299, 0.6791, 0.7239, 0.7638, 0.8310, 0.8624, 0.8944, 0.9241
])

cd_data = np.array([
    0.01365, 0.01367, 0.01370, 0.01376, 0.01384, 0.01395, 0.01408, 0.01424, 0.01443, 0.01465,
    0.01490, 0.01520, 0.01554, 0.01597, 0.01674, 0.01783, 0.02061, 0.02179, 0.02330, 0.02556
])


# Interpolation function
cd_interp = interp1d(alpha_data, cd_data, kind='linear', bounds_error=False, fill_value="extrapolate")
cl_interp = interp1d(alpha_data, cl_data, kind='linear', bounds_error=False, fill_value="extrapolate")

def xfoil_airfoil_CD(alpha):
    return float(cd_interp(alpha))

def xfoil_airfoil_CL(alpha):
    return float(cl_interp(alpha))