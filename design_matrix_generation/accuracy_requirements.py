## This script calculates the required number of measurement points for a given:
# - alpha-error
# - beta-error
# - minimum number of points (p, dependent on selection of polynomial order, cross-factors)
# - delta_certainty 


## Import required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

## Function: calculate number of points required
def n_points_required(alpha, beta, p, delta_certainty):
    """
    Calculate the number of points required for a given alpha, beta, p, and delta
    :param alpha: accepted probability of alpha-error
    :param beta: accepted probability of beta-error
    :param p: minimum number of points
    :param delta_certainty: delta as percentage of LSD
    :return: number of points required
    """
    
    # Calculate the z-value given the specified alpha
    z_alpha = np.abs(np.round(stats.norm.ppf(alpha/2), 2))
    
    # Calculate the z-value given the specified beta
    z_beta = np.abs(np.round(stats.norm.ppf(beta), 2))
    
    # Calculate the value of K given the specified value of delta
    # K = np.sqrt(2) * np.abs(np.round(stats.norm.ppf(delta_certainty/2), 2))    
    # Alternative: simply set K to example value
    K = 2 * np.sqrt(2)
    
    # Calculate the number of points required
    n = ((z_alpha + z_beta)/K)**2 * p
    
    ## Print the values of z_alpha, z_beta, and K
    print("For the given values of alpha, beta, p, and delta_certainty:")
    
    # Print the values of alpha, beta, p, and delta_tolerance
    print(f"alpha: {alpha}")
    print(f"beta: {beta}")
    print(f"p: {p}")
    print(f"delta_certainty: {delta_certainty}")
    
    print(f"z_alpha: {z_alpha}")
    print(f"z_beta: {z_beta}")
    print(f"K: {K}")
    print(f"Number of points required: {n}")
    
    ## Return the number of points required
    return n

## Run the function
n = n_points_required(0.05, 0.01, 6, 0.05)