import pandas as pd
import constants as c
import BoundaryCorrections as bc
import ResponseSurfaceModel as rsm
import Validation as val
import Plots as plt

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

# ---------------------------------------------------------------------
# Read the data from the uncorrected tail-off run
# ---------------------------------------------------------------------
df_less_tail_unc = pd.read_csv('tunnel_data_unc/uncorrected_tailoff.txt', delimiter='\t')

## ---------------------------------------------------------------------
## BEFORE CORRECTIONS: Support Load Subtraction
## ---------------------------------------------------------------------
# Support Load Subtraction: Subtract the strut loads from the uncorrected
# using the data provided for the model-less run.
# TODO: Might have to implement the support load subtraction after performing blockage correction.
df_less_model = pd.read_csv('model_off_data/modelOffData_condensed.csv', delimiter=',')
df_unc = bc.subtract_model_off_data(df_unc, df_less_model)
df_less_tail_unc = bc.subtract_model_off_data(df_less_tail_unc, df_less_model)
# TODO: CMpitch and CMpitch25c are now different, might need to also apply the correction to the 25c data

## ---------------------------------------------------------------------
## BEFORE CORRECTIONS: Thrust/Drag Accounting
## ---------------------------------------------------------------------
# Thrust/Drag Accounting: Calculate the thrust and update the drag coefficients
from thrust_drag_accounting.thrust_correction_df import thrust_correction
import pandas as pd

pd.set_option('display.max_columns', None)
df_unc = thrust_correction(df_unc)

# TODO: This has to be implemented using Remco's code.
# TODO: Thrust accounting depends on the effective velocity, which depends on the total blockage correction. Seems like we have a loop; should sort this out.
# TODO: Wake blockage correction depends on the drag coefficient, which depends on the thrust accounting. Could apply the solid blcokage correction before the thrust accounting.
# TODO: Could then iterate: thrust/drag accounting <-> wake/slipstream blockage correction
## ---------------------------------------------------------------------
## Apply the blockage corrections
## ---------------------------------------------------------------------
df = bc.apply_total_blockage_corrections(df_unc)

## ---------------------------------------------------------------------
## Apply the lift-interference corrections
## ---------------------------------------------------------------------
df = bc.apply_lift_interference_correction(df, df_less_tail_unc)

## ---------------------------------------------------------------------
## Write the corrected data to a file
## ---------------------------------------------------------------------
df.to_csv('tunnel_data_cor/corrected_data.txt', sep='\t', index=False)

## ---------------------------------------------------------------------
## Create the response surface model
## ---------------------------------------------------------------------

# TODO: create the response surface model

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
