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

saveallplots = True

for key in ['CL', 'CD', 'CMpitch']:

    RSM.plot_derivative_vs_alpha(save=saveallplots,key=key,derivative='alpha' , DELTA_E=[-10, 0,  10], J=1.6)
    RSM.plot_derivative_vs_alpha_J(save=saveallplots,key=key,derivative='alpha' , DELTA_E=[-10, 0,  10])

    

    std_tr = np.sqrt(RSM.training_loss[key])
    std_val = np.sqrt(RSM.validation_loss[key])
    mean_val = np.abs(RSM.validation_deltas[key].mean())

    # tolerance = 2 * np.sqrt(2) * stds_mean[key]  # BASED ON MEASUREMENT VARIANCE
    tolerance = 2 * np.sqrt(2) * std_tr  # BASED ON RSM VARIANCE

    # print(f'std of measurement {key}: {stds_mean[key]:.8f}')
    # # print(f'std of difference between validation set and model: {np.sqrt(RSM.validation_loss[key]):.8f}')
    # # print(f'std of difference between training set and model: {np.sqrt(RSM.training_loss[key]):.8f}')
    # print(f'model tolerance: {tolerance:.8f}')
    # print(f'std of validation set: {std_val:.8f}')
    # print(f'std of training set: {std_tr:.8f}')

    # print(f'mean of validation set: {mean_val:.8f}')

    RSM.significance_histogram(key, save=saveallplots)

    # prob_alpha = 1 - norm.cdf(std_val / std_tr)
    # prob_beta = norm.cdf((tolerance - std_val) / std_tr)

    # for validation_value in RSM.validation_deltas[key]:
        # print(f'Validation delta: {validation_value:.4f}, {validation_value / std_tr:.4f} std')
        # prob_alpha = (1 - norm.cdf(validation_value / std_tr)) 
        # prob_beta = (1 - norm.cdf((tolerance - validation_value) / std_tr))
        # # note: both of these should be high(er than the established bounds: 5% for alpha, 1% for beta)
        # # TODO: more math more gooder
        # print(f'Probability validation delta higher than recorded, assuming valid model: {prob_alpha * 100:.4f}%')
        # print(f'Probability validation delta lower than recorded, assuming invalid (biased) model: {prob_beta * 100:.4f}%')
        # print(f'likelihood of valid model: {prob_alpha / prob_beta:.4f}')

    # print('Probability of validation mean being further than recorded, assuming valid model: {:.4f}%'.format(100 * 2 * norm.cdf(-np.abs(mean_val) / std_tr  *  np.sqrt(len(RSM.validation_deltas[key])))))
    # print('Probability of ALL samples being further than recorded, assuming valid model and pairwise independent validation set:{:.4f}%'.format(100 * np.prod([2 * norm.cdf(-np.abs(validation_value) / std_tr) for validation_value in RSM.validation_deltas[key]])))
    # print('Probability of ALL samples being further than recorded maximum, assuming valid model:{:.4f}%'.format(100 * np.prod([2 * norm.cdf(-np.max(np.abs(validation_value)) / std_tr) for validation_value in RSM.validation_deltas[key]])))








