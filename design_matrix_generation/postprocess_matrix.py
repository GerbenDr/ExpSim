import pandas as pd
import numpy as np

points_per_block = 10 
de_setting_0 = 0  # starting setting
de_setting_1 = 10
obligatory_testpoint = np.array([8, 1.6, 0, 40]).T

V_base = 40
V_diff = 20

n_different_Re = 10

np.random.seed(0)

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
    pairs = main_block_one.reshape(4, main_block_one.shape[1] // 2, 2)  # Shape becomes (4, N//2, 2)
    pairs = shuffle_along_axis(pairs, axis=-1)
    main_block_one = pairs.reshape(4, main_block_one.shape[1] )

    pairs = main_block_two.reshape(4, main_block_two.shape[1]  // 2, 2)  # Shape becomes (4, N//2, 2)
    pairs = shuffle_along_axis(pairs, axis=-1)
    main_block_two = pairs.reshape(4, main_block_two.shape[1] )

    # # random permutation
    # permuted_columns = np.random.permutation(main_block_one.shape[1])
    # main_block_one = main_block_one[:, permuted_columns]

    # permuted_columns = np.random.permutation(main_block_two.shape[1])
    # main_block_two = main_block_two[:, permuted_columns]

    if not np.any([np.all(main_block_one[:, index] == obligatory_testpoint, axis=0) for index in range(main_block_one.shape[1])]):
        print('obligatory test point not included, appending an additional measurement point')
        main_block_one = np.concatenate((main_block_one, obligatory_testpoint), axis=1)


    wind_off_block = np.vstack(
        (
            np.random.permutation(unique_alphas), np.zeros(num_unique_alphas), np.full(num_unique_alphas, de_setting_0), np.zeros(num_unique_alphas)
        )
    )

    n_sweep = main_block_one.shape[1] // n_different_Re
    diff_re_block = main_block_one[:, ::n_sweep][:, :n_different_Re].copy()
    diff_re_block[3, :] = np.full(diff_re_block.shape[1], V_diff)
    
    full_matrix = np.concatenate(
        (wind_off_block, main_block_one, diff_re_block, main_block_two), axis=1
    )

    where_acoustic = np.where(
        np.logical_and(
            full_matrix[0, :] == obligatory_testpoint[0],
            full_matrix[2, :] == de_setting_0
        )
    )[0]

    # Initialize a boolean array of the same length as a, filled with False
    bool_array = np.zeros(full_matrix.shape[1], dtype=bool)

    # Set the indices where the condition is True to True
    bool_array[where_acoustic] = True

    full_matrix = np.vstack([full_matrix, bool_array])

    return full_matrix

if __name__ == "__main__":

    path = "design_matrix_generation/raw_test_matrix.csv"

    matrix = postprocess_matrix(path)

    print(matrix)

    # Convert the NumPy matrix to a DataFrame for easy handling
    df = pd.DataFrame(matrix.T, columns=["AoA", "J", "delta_a", "V_inf", "Acoustic"])

    # Save the DataFrame to a CSV file
    df.to_csv('matrix_output.csv', index=False)

    
    