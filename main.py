import pandas as pd
import constants as c
import BoundaryCorrections as bc
import ResponseSurfaceModel as rsm
import Validation as val
import Plots as plt
from thrust_drag_accounting.thrust_correction_df import thrust_correction

# ---------------------------------------------------------------------
## Combine the data from the minus 10 and plus 10 runs
## This is the data as outputted by the Matlab scripts (Lucille's code)
# ---------------------------------------------------------------------
# Read the data from the minus 10 run
df_minus_10 = pd.read_csv('tunnel_data_unc/uncorrected_elevator_minus_10.txt', delimiter='\t')

# Read the data from the plus 10 run
df_plus_10 = pd.read_csv('tunnel_data_unc/uncorrected_elevator10.txt', delimiter='\t')

# Combine the data and add a column for the elevator angle
df_minus_10['delta_e'] = -10
df_plus_10['delta_e'] = 10
df_unc = pd.concat([df_minus_10, df_plus_10])

# Save the combined data
df_unc.to_csv('tunnel_data_unc/combined_data.txt', sep='\t', index=False)

# Row 31 is missing data; for the integrity of the data, we will copy the values from row 30 to row 31
# Row 31 should be excluded from the masks
df_unc = df_unc.reset_index(drop=True)
df_unc.loc[31] = df_unc.loc[30]


# ---------------------------------------------------------------------
## Read the data from the model-off run
# ---------------------------------------------------------------------
df_less_model_unc = pd.read_csv('model_off_data/modelOffData_condensed.csv', delimiter = ',')

# ---------------------------------------------------------------------
# Read the data from the uncorrected tail-off run
# ---------------------------------------------------------------------
df_less_tail_unc = pd.read_csv('tunnel_data_unc/uncorrected_tailoff.txt', delimiter='\t')

## ---------------------------------------------------------------------
## BEFORE CORRECTIONS: Thrust/Drag Accounting
## ---------------------------------------------------------------------
# Thrust/Drag Accounting: Calculate the thrust force from the uncorrected data.
df_thrust_cor = thrust_correction(df_unc)

# Calculating the measured force in "drag-direction"
df_thrust_cor['F_meas'] = df_thrust_cor['q'] * c.S_REF * df_thrust_cor['CD']

# Calculating the drag force by subtracting the calculated thrust from the measured force in 'drag-direction'
df_thrust_cor['Drag_Force'] = df_thrust_cor['F_meas'] + df_thrust_cor['thrust']

# Calculating the drag coefficient corrected for the thrust
df_thrust_cor['CD_thrust_cor'] = df_thrust_cor['Drag_Force'] / (df_thrust_cor['q'] * c.S_REF)

## ---------------------------------------------------------------------
## BEFORE CORRECTIONS: Support Load Subtraction
## ---------------------------------------------------------------------
# Support Load Subtraction: Subtract the strut loads from the uncorrected data using the model-less data.
df_sup_cor = bc.subtract_model_off_data(df_thrust_cor, df_less_model_unc, tail_on=True)
df_less_tail_unc = bc.subtract_model_off_data(df_less_tail_unc, df_less_model_unc, tail_on=False)

## ---------------------------------------------------------------------
## Apply the blockage corrections to our measurement data
# Solid blockage, wake blockage and slipstream blockage are included.
## ---------------------------------------------------------------------
df_block_cor = bc.apply_total_blockage_corrections(df_sup_cor)

## ---------------------------------------------------------------------
## Apply the lift interference corrections to our measurement data
## ---------------------------------------------------------------------
# First, apply the blockage corrections to the tail-less data
df_less_tail = bc.apply_total_blockage_corrections(df_less_tail_unc, tail_on=False)
df_less_tail_cor = bc.apply_lift_interference_correction_less_tail(df_less_tail)

# Then apply the lift interference corrections to the measurements using the tail-less data
df_lift_cor = bc.apply_lift_interference_correction(df_block_cor, df_less_tail_cor)

## ---------------------------------------------------------------------
## Write the corrected data to a file
## ---------------------------------------------------------------------
df_lift_cor.to_csv('tunnel_data_cor/corrected_data.txt', sep='\t', index=False)
df = df_lift_cor

## ---------------------------------------------------------------------
## Create the response surface model
## ---------------------------------------------------------------------
df_RSM = df.loc[c.mask_RSM]
df_validation = df.loc[c.mask_validation]

rsm_instance = rsm.ResponseSurfaceModel(df_RSM, df_validation)

coeff, res, loss, deltas = rsm_instance.fit()
print(coeff, res, loss, deltas)

## ---------------------------------------------------------------------
## Extract relevant parameters from response surface model
## ---------------------------------------------------------------------
# TODO

## ---------------------------------------------------------------------
## Calculate standard deviation of measurements
## ---------------------------------------------------------------------
# TODO

## ---------------------------------------------------------------------
## Validation Points
## ---------------------------------------------------------------------
# TODO

## ---------------------------------------------------------------------
## Create relevant plots
## ---------------------------------------------------------------------
# TODO
