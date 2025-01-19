import pandas as pd
import numpy as np
from itertools import product

# points_per_block = 10
de_setting_0 = -10  # starting setting
de_setting_1 = 10
obligatory_testpoints = [np.array([8, 1.6, de_setting_0, 40]).reshape(4, 1), np.array([8, 0, de_setting_0, 40]).reshape(4, 1)]

V_base = 40
V_diff = 20

n_different_Re = 8

N_validation = 2 # per corner point

alpha_range = (-4, 7, 0.5)
J_range = (1.6, 2.4, 0.1)
deltae_range = (-10, 10, 10)

# Generate the corner points
corner_points = []
for r in [alpha_range, J_range, deltae_range]:
    corner_points.append([r[0], r[1]])

# Create all combinations of corner points
corner_combinations = np.array(list(product(*corner_points)))

np.random.seed(1)

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

def postprocess_matrix(path):

    df = pd.read_csv(path)

    J = df["design.J"]
    alpha = df["design.alpha"]
    delta_e = df["design.delta_e"]
    V_inf = np.full(J.shape, V_base)

    unique_alphas = np.unique(alpha)
    num_unique_alphas = len(unique_alphas)

    de_0 = np.where(delta_e == de_setting_0)[0]
    de_1 = np.where(delta_e == de_setting_1)[0]

    main_block_one = np.vstack(
        (
            alpha[de_0], J[de_0], delta_e[de_0], V_inf[de_0]
        )
    )

    main_block_two = np.vstack(
        (
            alpha[de_1], J[de_1], delta_e[de_1], V_inf[de_1]
        )
    )

    # pair-wise randomization
    # pairs = main_block_one.reshape(4, main_block_one.shape[1] // 2, 2)  # Shape becomes (4, N//2, 2)
    # permuted_pairs = pairs[:, np.random.permutation(pairs.shape[1]), :]
    # main_block_one = permuted_pairs.reshape(4, main_block_one.shape[1])

    # pairs = main_block_two.reshape(4, main_block_two.shape[1] // 2, 2)  # Shape becomes (4, N//2, 2)
    # permuted_pairs = pairs[:, np.random.permutation(pairs.shape[1]), :]
    # main_block_two = permuted_pairs.reshape(4, main_block_two.shape[1])

    # # random permutation
    permuted_columns = np.random.permutation(main_block_one.shape[1])
    main_block_one = main_block_one[:, permuted_columns]

    permuted_columns = np.random.permutation(main_block_two.shape[1])
    main_block_two = main_block_two[:, permuted_columns]


    for obligatory_testpoint in obligatory_testpoints:
        if not np.any([np.all(main_block_one[:, index] == obligatory_testpoint.T) for index in range(main_block_one.shape[1])]):
            print('obligatory test point not included, appending an additional measurement point')
            main_block_one = np.concatenate((main_block_one, obligatory_testpoint), axis=1)

    # validation points
    # alpha_v = np.random.choice(np.arange(alpha_range[0], alpha_range[1] + alpha_range[2], alpha_range[2]), size=N_validation)
    # J_v = np.random.choice(np.arange(J_range[0], J_range[1] + J_range[2], J_range[2]), size=N_validation)
    # delta_e_v = np.random.choice(np.arange(deltae_range[0], deltae_range[1] + deltae_range[2], deltae_range[2]), size=N_validation)
    # V_v = np.full(N_validation, V_base)
    # if np.all(delta_e_v == de_setting_0) or np.all(delta_e_v == de_setting_1):
    #     print("WARNING: only one elevator setting in validation set")
    # validation_points = np.vstack((alpha_v, J_v, delta_e_v, V_v))
    # vp_b1 = validation_points[:, np.where(validation_points[2, :] == de_setting_0)[0]]
    # main_block_one = np.concatenate((main_block_one, vp_b1), axis=1)
    # vp_b2 = validation_points[:, np.where(validation_points[2, :] == de_setting_1)[0]]
    # main_block_two = np.concatenate((main_block_two, vp_b2), axis=1)


    indices_to_duplicate = np.where(

        np.any(tuple(np.all(np.equal(main_block_one[:3, :], np.tile(corner_combinations[i], reps=(main_block_one.shape[1],1)).T), axis=0) for i in range(len(corner_combinations))), axis=0)
    )[0]
    indices_to_duplicate.sort()

    for k in indices_to_duplicate:
        column_to_duplicate = main_block_one[:, k]
        # Create two copies of the column
        duplicated_columns = np.tile(column_to_duplicate[:, np.newaxis], (1, N_validation))
        # Insert the duplicated columns right after the k'th column
        main_block_one = np.insert(main_block_one, k + 1, duplicated_columns.T, axis=1)
        indices_to_duplicate += N_validation


    indices_to_duplicate = np.where(

        np.any(tuple(np.all(np.equal(main_block_two[:3, :], np.tile(corner_combinations[i], reps=(main_block_two.shape[1],1)).T), axis=0) for i in range(len(corner_combinations))), axis=0)
    )[0]
    indices_to_duplicate.sort()

    for k in indices_to_duplicate:
        column_to_duplicate = main_block_two[:, k]
        # Create two copies of the column
        duplicated_columns = np.tile(column_to_duplicate[:, np.newaxis], (1, N_validation))
        # Insert the duplicated columns right after the k'th column
        main_block_two = np.insert(main_block_two, k + 1, duplicated_columns.T, axis=1)
        indices_to_duplicate += N_validation

    wind_off_block = np.vstack(
        (
            np.random.permutation(unique_alphas), np.zeros(num_unique_alphas), np.full(num_unique_alphas, de_setting_0), np.zeros(num_unique_alphas)
        )
    )

    # n_sweep = main_block_one.shape[1] // n_different_Re
    # diff_re_block = main_block_one[:, ::n_sweep][:, :n_different_Re].copy()
    # diff_re_block[3, :] = np.full(diff_re_block.shape[1], V_diff)

    # alpha_Re = np.linspace(alpha_range[0], alpha_range[1], n_different_Re)
    # J_Re = np.linspace(J_range[0], J_range[1] , n_different_Re)
    alpha_Re = np.random.choice(np.arange(alpha_range[0], alpha_range[1] + alpha_range[2], alpha_range[2]), size=n_different_Re)
    J_Re = np.random.choice(np.arange(J_range[0], J_range[1] + J_range[2], J_range[2]), size=n_different_Re)

    delta_e_Re =np.full(n_different_Re, de_setting_0)
    V_Re = np.full(n_different_Re, V_diff)

    diff_re_block = np.vstack((alpha_Re, J_Re, delta_e_Re, V_Re))

    # random permutation
    permuted_columns = np.random.permutation(diff_re_block.shape[1])
    diff_re_block = diff_re_block[:, permuted_columns]

    full_matrix = np.concatenate(
        (wind_off_block, main_block_one, diff_re_block, main_block_two), axis=1
    )

    alpha_acoustic = np.max(unique_alphas)
    where_acoustic = np.where(
        np.logical_and(
        np.logical_or(
            np.logical_and(
                full_matrix[0, :] == alpha_acoustic,
                full_matrix[2, :] == de_setting_0
            ),
            np.any(tuple(np.all(np.equal(full_matrix, obligatory_testpoints[i]), axis=0) for i in range(len(obligatory_testpoints))), axis=0)
        ),
        full_matrix[3, :] != 0
        )
    )[0]

    # Initialize a boolean array of the same length as a, filled with False
    bool_array = np.zeros(full_matrix.shape[1], dtype=bool)

    # Set the indices where the condition is True to True
    bool_array[where_acoustic] = True

    full_matrix = np.vstack([full_matrix, bool_array])

    return full_matrix

if __name__ == "__main__":
    # uncomment when using pycharm
    # import os

    # os.chdir('..')

    path = "design_matrix_generation/raw_test_matrix.csv"

    matrix = postprocess_matrix(path)

    print(matrix)

    # Convert the NumPy matrix to a DataFrame for easy handling
    df = pd.DataFrame(matrix.T, columns=["AoA", "J", "delta_a", "V_inf", "Acoustic"])

    # Save the DataFrame to a CSV file
    df.to_csv('matrix_output.csv', index=False)

    
    