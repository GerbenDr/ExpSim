import pandas as pd
import constants as c
import BoundaryCorrections as bc

# ---------------------------------------------------------------------
## Combine the data from the minus 10 and plus 10 runs
## This is the data as outputted by the Matlab scripts (Lucille's code)
# ---------------------------------------------------------------------
# Read the data from the minus 10 run
df_minus_10 = pd.read_csv('tunnel_data/uncorrected_elevator_minus_10', delimiter = '\t')

# Read the data from the plus 10 run
df_plus_10 = pd.read_csv('tunnel_data/uncorrected_elevator_10', delimiter = '\t')

# Combine the data and add a column for the elevator angle
df_minus_10['delta_e'] = -10
df_plus_10['delta_e'] = 10
df_unc = pd.concat([df_minus_10, df_plus_10])

# Save the combined data
df_unc.to_csv('tunnel_data/combined_data.txt', sep = '\t', index = False)

# ---------------------------------------------------------------------
# TODO: If we need to "filter" the uncorrected data more, we can do this here...
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Read the data from the uncorrected tail-off run
# ---------------------------------------------------------------------
df_less_tail_unc = pd.read_csv('tunnel_data/uncorrected_tailoff.txt', delimiter = '\t')


## ---------------------------------------------------------------------
## BEFORE CORRECTIONS: Thrust/Drag Accounting, Subtracting Support Loads
## ---------------------------------------------------------------------
# Thrust/Drag Accounting: Calculate the thrust and update the drag coefficients
# TODO: This has to be implemented using Remco's code.

# Support Load Subtraction: Subtract the strut loads from the uncorrected
# using the data provided for the model-less run.
# TODO: Import correct data from the model-less run and implement this.


## ---------------------------------------------------------------------
## Apply the blockage corrections
## ---------------------------------------------------------------------
df = bc.apply_total_blockage_corrections(df_unc)

## ---------------------------------------------------------------------
## Apply the lift-interference corrections
## ---------------------------------------------------------------------
df = bc.apply_lift_interference_corrections(df, df_less_tail_unc)








