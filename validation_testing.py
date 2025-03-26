import pandas as pd
import constants as c
import BoundaryCorrections as bc
import ResponseSurfaceModel as rsm
import Validation as val
import Plots as plt
import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------------------
## Combine the data from the minus 10 and plus 10 runs
## This is the data as outputted by the Matlab scripts (Lucille's code)
# ---------------------------------------------------------------------
# Read the data from the minus 10 run
df_minus_10 = pd.read_csv('tunnel_data_unc/uncorrected_elevator_minus_10.txt', delimiter = '\t')

# Read the data from the plus 10 run
df_plus_10 = pd.read_csv('tunnel_data_unc/uncorrected_elevator10.txt', delimiter = '\t')

# Combine the data and add a column for the elevator angle
df_minus_10['delta_e'] = -10
df_plus_10['delta_e'] = 10
df_unc = pd.concat([df_minus_10, df_plus_10])

# Save the combined data
df_unc.to_csv('tunnel_data_unc/combined_data.txt', sep = '\t', index = False)

df_less_model = pd.read_csv('model_off_data/modelOffData_condensed.csv', delimiter = ',')
df_unc = bc.subtract_model_off_data(df_unc, df_less_model)

df_RSM = df_unc.loc[c.mask_RSM]
df_validation = df_unc.loc[c.mask_validation]

N_repetition_points = len(c.mask_repetition_pointwise_inclusive)

stds = [0]*N_repetition_points
means = [0]*N_repetition_points

for i, mask in enumerate(c.mask_repetition_pointwise_inclusive):
    df_repetition = df_unc.loc[mask]
    stds[i] = df_repetition.std(ddof=1) 
    means[i] = df_repetition.mean()

    # for key in ['CL', 'CD', 'CMpitch']:
    #     print(key, stds[i][key] / means[i][key])


stds_mean = sum(stds) / N_repetition_points

RSM = rsm.ResponseSurfaceModel(df_RSM, df_validation)

# print(np.trace(RSM.coefficient_covariance))
# print(np.trace(RSM.prediction_covariance))
RSM.print_hypothesis_test_results()

saveallplots = False

for key in ['CL', 'CD', 'CMpitch']:

    RSM.plot_RSM_1D(save=saveallplots, key=key, J=1.8, DELTA_E= -10, reference_dataframe='self', validation_dataframe='self')
    # RSM.plot_derivative_vs_alpha(save=saveallplots,key=key,derivative='alpha' , DELTA_E=[-10, 0,  10], J=1.6)
    # RSM.plot_derivative_vs_alpha_J(save=saveallplots,key=key,derivative='alpha' , DELTA_E=[-10, 0,  10])
    # RSM.significance_histogram(key, save=saveallplots)




