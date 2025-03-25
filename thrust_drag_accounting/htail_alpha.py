import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Path to the CSV file
file_path = 'yplus_14_komegasst.csv'

# Read the CSV into a DataFrame
df = pd.read_csv(file_path, delimiter=',')

def calculate_freestream_alpha(df):
    df['freestream_alpha'] = -np.degrees(np.arctan(df['P3'] / df['P1']))
    return df

def calculate_htail_alpha(df):
    df['htail_alpha'] = -np.degrees(np.arctan(df['P5'] / df['P4']))
    return df

def add_velocity_magnitude(df):
    df['velocity_magnitude'] = np.sqrt(df['P5']**2 + df['P4']**2)
    return df

def interpolate_htail_alpha(df):
    # Create interpolation function
    interp_func = interp1d(df['freestream_alpha'], df['htail_alpha'], kind='linear', fill_value='extrapolate')
    return interp_func

def interpolate_velocity_magnitude(df):
    interp_func = interp1d(df['freestream_alpha'], df['velocity_magnitude'], kind='linear', fill_value='extrapolate')
    return interp_func

df = calculate_freestream_alpha(df)
df = calculate_htail_alpha(df)
df = add_velocity_magnitude(df)

print(df)
htail_alpha = interpolate_htail_alpha(df)
htail_velocity = interpolate_velocity_magnitude(df)