import pandas as pd
import constants as c
import BoundaryCorrections as bc
import ResponseSurfaceModel as rsm
import Validation as val
import Plots as plt
import numpy as np

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

RSM = rsm.ResponseSurfaceModel(df_RSM)

result, residuals, loss = RSM.predict(df_validation)

print(loss, RSM.training_loss)
