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

# Ditch row 31, which is incomplete
df_unc = df_unc.drop(31)

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
df_unc = thrust_correction(df_unc)

# Calculating the measured force in "drag-direction"
df_unc['F_meas'] = df_unc['q'] * c.S_REF * df_unc['CD']

# Calculating the drag force by subtracting the calculated thrust from the measured force in 'drag-direction'
df_unc['Drag_Force'] = df_unc['F_meas'] + df_unc['thrust']

# Calculating the drag coefficient corrected for the thrust
df_unc['CD_thrust_cor'] = df_unc['Drag_Force'] / (df_unc['q'] * c.S_REF)

## ---------------------------------------------------------------------
## BEFORE CORRECTIONS: Support Load Subtraction
## ---------------------------------------------------------------------
# Support Load Subtraction: Subtract the strut loads from the uncorrected data using the model-less data.
df_unc = bc.subtract_model_off_data(df_unc, df_less_model_unc, tail_on=True)
df_less_tail_unc = bc.subtract_model_off_data(df_less_tail_unc, df_less_model_unc, tail_on=False)

## ---------------------------------------------------------------------
## Apply the blockage corrections to our measurement data
# Solid blockage, wake blockage and slipstream blockage are included.
## ---------------------------------------------------------------------
df = bc.apply_total_blockage_corrections(df_unc)

## ---------------------------------------------------------------------
## Apply the lift interference corrections to our measurement data
## ---------------------------------------------------------------------
# First, apply the blockage corrections to the tail-less data
df_less_tail = bc.apply_total_blockage_corrections(df_less_tail_unc, tail_on=False)
df_less_tail_cor = bc.apply_lift_interference_correction_less_tail(df_less_tail)

# Then apply the lift interference corrections to the measurements using the tail-less data
df = bc.apply_lift_interference_correction(df, df_less_tail_cor)

## ---------------------------------------------------------------------
## Write the corrected data to a file
## ---------------------------------------------------------------------
df.to_csv('tunnel_data_cor/corrected_data.txt', sep='\t', index=False)

## ---------------------------------------------------------------------
## Create the response surface model
## ---------------------------------------------------------------------
rsm_instance = rsm.ResponseSurfaceModel(df)
coeff, res = rsm_instance.fit()
print(coeff, res)

for key in ['CL', 'CD', 'CMpitch']:
    rsm.plot_RSM(key=key, DELTA_E=-10, save=True, reference_dataframe='self')
    rsm.plot_RSM(key=key, AOA=7, save=True, reference_dataframe='self')
    rsm.plot_RSM(key=key, J=1.8, save=True, reference_dataframe='self')

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
