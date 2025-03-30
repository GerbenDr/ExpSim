import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path

def calculate_freestream_alpha(df):
    df['freestream_alpha'] = -np.degrees(np.arctan(df['P3'] / df['P1']))
    return df

def calculate_htail_alpha(df):
    df['htail_alpha'] = -np.degrees(np.arctan(df['P5'] / df['P4']))
    return df

def add_velocity_magnitude(df):
    df['velocity_magnitude'] = np.sqrt(df['P5']**2 + df['P4']**2)
    return df

def process_file(filename):
    file_path = Path(__file__).resolve().parent / filename
    df = pd.read_csv(file_path, delimiter=',')
    df = calculate_freestream_alpha(df)
    df = calculate_htail_alpha(df)
    df = add_velocity_magnitude(df)
    return df

# Load the datasets for 20 m/s and 40 m/s freestream velocities
df_20 = process_file("RSM_yplus_5_20ms.csv")
df_40 = process_file("RSM_yplus_5_40ms.csv")

# Create interpolation functions for each dataset
interp_htail_alpha_20 = interp1d(df_20['freestream_alpha'], df_20['htail_alpha'], kind='linear', fill_value='extrapolate')
interp_velocity_20   = interp1d(df_20['freestream_alpha'], df_20['velocity_magnitude'], kind='linear', fill_value='extrapolate')

interp_htail_alpha_40 = interp1d(df_40['freestream_alpha'], df_40['htail_alpha'], kind='linear', fill_value='extrapolate')
interp_velocity_40   = interp1d(df_40['freestream_alpha'], df_40['velocity_magnitude'], kind='linear', fill_value='extrapolate')

def htail_alpha(freestream_alpha, freestream_velocity):
    """
    Return the horizontal tail angle for a given freestream alpha and velocity.
    Extrapolates linearly for freestream velocities slightly outside the 20-40 m/s range.
    """
    alpha_20 = float(interp_htail_alpha_20(freestream_alpha))
    alpha_40 = float(interp_htail_alpha_40(freestream_alpha))
    weight = (freestream_velocity - 20) / 20.0  # weight can be < 0 or > 1 for extrapolation
    return alpha_20 + weight * (alpha_40 - alpha_20)

def htail_velocity(freestream_alpha, freestream_velocity):
    """
    Return the horizontal tail velocity magnitude for a given freestream alpha and velocity.
    Extrapolates linearly for freestream velocities slightly outside the 20-40 m/s range.
    """
    vel_20 = float(interp_velocity_20(freestream_alpha))
    vel_40 = float(interp_velocity_40(freestream_alpha))
    weight = (freestream_velocity - 20) / 20.0  # weight can be < 0 or > 1 for extrapolation
    return vel_20 + weight * (vel_40 - vel_20)