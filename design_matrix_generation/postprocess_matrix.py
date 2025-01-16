import pandas as pd
import numpy as np

points_per_block = 10 
de_setting_0 = 0  # starting setting
de_setting_1 = 10
obligatory_testpoint = np.array([8, 1.6, 0, 40]).T

V_base = 40
V_diff = 20

n_different_Re = 20

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

    if not np.any([np.all(main_block_one[:, index] == obligatory_testpoint, axis=0) for index in range(main_block_one.shape[1])]):
        print('obligatory test point not included, appending an additional measurement point')
        main_block_one = np.concatenate((main_block_one, obligatory_testpoint), axis=1)


    wind_off_block = np.vstack(
        (
            unique_alphas, np.zeros(num_unique_alphas), np.full(num_unique_alphas, de_setting_0), np.zeros(num_unique_alphas)
        )
    )

    diff_re_block = main_block_one[:, :n_different_Re].copy()
    diff_re_block[3, :] = np.full(n_different_Re, V_diff)
    
    full_matrix = np.concatenate(
        (wind_off_block, main_block_one, diff_re_block, main_block_two), axis=1
    )

    where_acoustic = np.where(
        np.logical_and(
            full_matrix[0, :] == obligatory_testpoint[0],
            full_matrix[2, :] == de_setting_0
        )
    )[0]

    return full_matrix, where_acoustic

if __name__ == "__main__":

    path = "design_matrix_generation/raw_test_matrix.csv"

    matrix, where_acoustic = postprocess_matrix(path)

    print(matrix)
    print(where_acoustic)
    print(where_acoustic.shape)
    print(matrix[:, where_acoustic])
    