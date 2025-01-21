

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def get_validation_points(n_validation_points, plot=False):
    # Load the data
    data_file = "design_matrix_generation/raw_test_matrix.csv"  # Replace with your file name
    data = pd.read_csv(data_file)

    # Extract relevant columns
    alpha = data['design.alpha']
    J = data['design.J']
    delta_e = data['design.delta_e']

    # Get min and max values for each column
    alpha_min, alpha_max = alpha.min(), alpha.max()
    J_min, J_max = J.min(), J.max()
    delta_e_min, delta_e_max = delta_e.min(), delta_e.max()

    # Specify the range which is the difference between the max and min values
    alpha_range = alpha_max - alpha_min
    J_range = J_max - J_min
    delta_e_range = delta_e_max - delta_e_min

    # Print the minimum and maximum values
    print(f"Alpha: min={alpha_min:.3f}, max={alpha_max:.3f}")
    print(f"J: min={J_min:.3f}, max={J_max:.3f}")
    print(f"Delta_e: min={delta_e_min:.3f}, max={delta_e_max:.3f}")

    # Create a grid of points for each delta_e value. Distribute alpha and J based on specified intervals
    alpha_interval = 0.5  # Interval for alpha
    J_interval = 0.1  # Interval for J

    alpha_grid = np.arange(alpha_min, alpha_max + alpha_interval, alpha_interval)
    J_grid = np.arange(J_min, J_max + J_interval, J_interval)

    # Generate grid points for all combinations of alpha, J, and delta_e
    alpha_grid, J_grid, delta_e_grid = np.meshgrid(alpha_grid, J_grid, [delta_e_min, delta_e_max])
    grid_points = np.column_stack((alpha_grid.ravel(), J_grid.ravel(), delta_e_grid.ravel()))

    print(f"Grid Points: {grid_points}")

    # Original points
    original_points = data[['design.alpha', 'design.J', 'design.delta_e']].values
    print(f"Original Points: {original_points}")

    # Function to find the furthest point
    def find_furthest_point(grid_points, alpha_range, J_range, delta_e_range):
        min_distances = []

        for i, grid_point in enumerate(grid_points):
            distances = []
            
            for j, original_point in enumerate(original_points):
                distance = np.sqrt(((grid_point[0] - original_point[0])/alpha_range)**2 + ((grid_point[1] - original_point[1])/J_range)**2 + ((grid_point[2] - original_point[2])/delta_e_range)**2)  
                distances.append(distance)
                
            minimum_distance = np.min(distances)
            min_distances.append([minimum_distance, grid_points[i]])

        max_distance = 0
        furthest_point = None
        for distance in min_distances:
            if distance[0] > max_distance:
                max_distance = distance[0]
                furthest_point = distance[1]
        
        return furthest_point

    # Find and add the furthest point multiple times
    furthest_points = []

    for _ in range(n_validation_points):
        furthest_point = find_furthest_point(grid_points, alpha_range, J_range, delta_e_range)
        furthest_points.append(furthest_point)
        original_points = np.vstack([original_points, [furthest_point[0], furthest_point[1], furthest_point[2]]])
        
    if plot:
        plot_design_matrix_and_validation(furthest_points, data)
        
    return furthest_points

def plot_design_matrix_and_validation(furthest_points, data):
    # Plot the point with the other points in the design matrix. Plot the furthest point in red, otheres in blue
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot original points
    ax.scatter(data['design.alpha'], data['design.J'], data['design.delta_e'], color='blue', label='Design Points')

    # Plot the tested grid poitns
    # ax.scatter(grid_points[:, 0], grid_points[:, 1], grid_points[:, 2], color='green', label='Grid Points')

    # Plot furthest points
    furthest_points = np.array(furthest_points)
    ax.scatter(furthest_points[:, 0], furthest_points[:, 1], furthest_points[:, 2], color='red', label='Furthest Points')

    ax.set_xlabel('Alpha')
    ax.set_ylabel('J')
    ax.set_zlabel('Delta_e')
    ax.legend()

    plt.show()
    
if __name__ == "__main__":
    n_validation_points = 4
    furthest_points = get_validation_points(n_validation_points, True)
    print(f"Furthest Points: {furthest_points}")