from postprocess_matrix import *

t_start_tunnel = 5 * 60
t_change_freestream = 60 * 1.5
t_small_change_freestream = 20
alpha_rate = 0.5  # degrees per second
t_change_prop = 15
t_change_elevator = 10 * 60
t_aerodynamic_measurement = 10
t_aeroacoustic_measurement_highpower = 10
t_aeroacoustic_measurement_lowpower = 20
t_human_operator = 10

total_tunnel_time = 60 * 60 * 1.75  # 1:45 minutes


def timings(matrix):
    """
    compute timing of experiment
    return: a numpy array of matrix, appended with a row for predicted wall-clock time
    """

    current_time = 0.0

    time = np.zeros(shape=matrix.shape[1])

    for index in range(matrix.shape[1]):

        if matrix[4, index] == 1:
            if matrix[3, index] == 40:
                t_measurement = t_aeroacoustic_measurement_highpower
            else:
                t_measurement = t_aeroacoustic_measurement_lowpower
        else:
            t_measurement = t_aerodynamic_measurement

        t_change_settings = t_human_operator

        if index == 0: # first measurement, only worry about alpha change time
            delta_alpha = matrix[0, index]
            t_change_settings += np.abs(delta_alpha) / alpha_rate

        else: # any subsequent measurement
            delta_alpha = matrix[0, index] - matrix[0, index-1]
            delta_J = matrix[1, index] - matrix[1, index-1]
            delta_delta_e = matrix[2, index] - matrix[2, index-1]
            delta_V_inf = matrix[3, index] - matrix[3, index-1]

            t_change_settings += np.abs(delta_alpha) / alpha_rate

            if delta_V_inf != 0:  # time to change free-stream conditions
                t_change_settings += t_change_freestream
            elif (delta_alpha != 0 or delta_J != 0) and matrix[3, index] != 0:
                t_change_settings += t_small_change_freestream

            if delta_J != 0: # time to adjust propeller RPM
                t_change_settings += t_change_prop

            if matrix[3, index-1] == 0 and matrix[3, index] != 0:  # time to start the tunnel
                t_change_settings += t_start_tunnel

            if delta_delta_e !=0:  # time to change elevator
                t_change_settings += t_change_elevator 

        t_tot = t_measurement + t_change_settings

        # store THE END TIME OF THE MEASUREMENT
        current_time += t_tot
        time[index] = current_time

    print("final measurement finished at: {:.2f} minutes".format(time[-1]  / 60))
    print("safety margin: {:.2f} minutes".format((total_tunnel_time - time[-1]) / 60))

    return np.vstack((matrix, time))

if __name__ == "__main__":

    # uncomment when using pycharm
    import os
    os.chdir('..')

    path = "design_matrix_generation/raw_test_matrix.csv"

    matrix = postprocess_matrix(path)
    matrix_w_timings = timings(matrix)

    # Convert the NumPy matrix to a DataFrame for easy handling
    df = pd.DataFrame(matrix_w_timings.T, columns=["AoA", "J", "delta_a", "V_inf", "Acoustic", "Time"])

    # Save the DataFrame to a CSV file
    df.to_csv('matrix_w_time.csv', index=False)



