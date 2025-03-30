import sys
from pathlib import Path

# Add root project directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from thrust_drag_accounting.prop_off_RSM import rsm_CD_po
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from thrust_drag_accounting.prop_off_RSM import rsm_CD_po  # same imports you have in the original code
import constants
from thrust_drag_accounting.airfoil_CD import xfoil_airfoil_CD, xfoil_airfoil_CL
from thrust_drag_accounting.htail_alpha import htail_alpha, htail_velocity




def read_data_to_df(file_path):
    """Reads the TSV file into a Pandas DataFrame."""
    df = pd.read_csv(file_path, delimiter='\t')
    return df


def calculate_T0(df):
    """
    Calculates and stores the initial thrust estimate (T_0) in a new column 'T_0',
    mirroring the 'calculate_T0' method in your original code.
    """
    # For each row, we use the predicted CD from rsm_CD_po
    # Then compute the difference in Fx to get T_0
    T0_values = []
    for idx, row in df.iterrows():
        # Predict the prop-off CD
        predicted_CD_po = rsm_CD_po.predict([[row['AoA'], row['V']]])[0]

        # Compute Fx for the predicted prop-off condition
        Fx_po = -predicted_CD_po * constants.S_REF * row['q']

        # Compute Fx from the actual measured coefficient
        Fx_pon = -row['CD'] * constants.S_REF * row['q']

        # T_0 is the difference
        T_0 = Fx_pon - Fx_po
        T0_values.append(T_0)
    df['T_0'] = T0_values
    return df


def thrust_iteration(row, current_thrust):
    """
    One iteration of thrust correction for a single row,
    similar to 'thrust_iteration' in the original code.
    """
    # Induced velocity guess
    alpha = row['AoA']
    rho = row['rho']
    freestream_velocity = row['V']
    htail_alpha_mo = htail_alpha(alpha, freestream_velocity)
    htail_velocity_mo = htail_velocity(alpha, freestream_velocity)
    v_i = 0.5 * np.sqrt(row[ 'V']**2 + (2 * current_thrust) / (constants.S_PROP * row['rho'])) - row['V'] / 2
    # r_wake:
    r_wake = constants.D_PROP*0.5*np.sqrt((row['V']+v_i)/(row['V'] + 2 * v_i))
    b_half_wake = 2*r_wake
    delta_e = row['delta_e']

    # Drag with original velocity
    # find lift and drag:
    D_motor_off = calculate_blown_area_drag(alpha, rho, htail_alpha_mo, htail_velocity_mo, b_half_wake, delta_e)

    # Drag with velocity + induced velocity
    htail_velocity_motor_on = np.sqrt((np.cos(np.deg2rad(htail_alpha_mo)) * htail_velocity_mo + v_i)**2 + (np.sin(np.deg2rad(htail_alpha_mo)) * htail_velocity_mo)**2)
    htail_alpha_motor_on = np.rad2deg(np.arcsin(np.sin(np.deg2rad(htail_alpha_mo)) * htail_velocity_mo/htail_velocity_motor_on))
    D_motor_on = calculate_blown_area_drag(alpha, rho, htail_alpha_motor_on, htail_velocity_motor_on, b_half_wake, delta_e)
    # print('drag, motor off and on:', D_motor_off, D_motor_on)

    delta_D = 2*(D_motor_on - D_motor_off) # the 2 is there to account for both motors!

    return float(row['T_0']) + delta_D

def calculate_blown_area_drag(alpha, rho, htail_alpha, htail_velocity, b_half_wake, delta_e):
    q_mo = 0.5 * rho * (htail_velocity ** 2)
    L_local = q_mo * xfoil_airfoil_CL(htail_alpha, delta_e) * b_half_wake * constants.C_HTAIL
    D_local = q_mo * xfoil_airfoil_CD(htail_alpha, delta_e) * b_half_wake * constants.C_HTAIL
    L_global, D_global = calculate_global_L_D(L_local, D_local, htail_alpha, alpha)
    # print('D_local vs global:', D_local, D_global)
    # print('L_local vs global:', L_local, L_global)
    return D_global

def calculate_global_L_D(L_local_mo, D_local_mo, alpha_local, alpha):
    D = np.sin(np.deg2rad(alpha - alpha_local)) * L_local_mo + np.cos(np.deg2rad(alpha - alpha_local)) * D_local_mo
    L = np.cos(np.deg2rad(alpha - alpha_local)) * L_local_mo - np.sin(np.deg2rad(alpha - alpha_local)) * D_local_mo
    return L, D


def thrust_correction(df, tol=1e-12, max_iter=100):
    """
    Performs the full thrust correction for each row in the DataFrame.
    After convergence, writes the final thrust value into a new column 'thrust'.
    """
    df = calculate_T0(df)

    final_thrusts = []

    for idx, row in df.iterrows():
        thrust = float(row['T_0'])
        try:
            for _ in range(max_iter):
                new_thrust = thrust_iteration(row, thrust)
                if abs(new_thrust - thrust) < tol:
                    break
                thrust = new_thrust

            if np.isnan(thrust):
                raise ValueError
        except ValueError:
            print(f"NaN encountered in thrust iteration at row {idx}, using T_0 instead.")
            thrust = float(row['T_0'])

        final_thrusts.append(thrust)

    df['thrust'] = final_thrusts
    return df


def plot_thrust_iteration_example(row, tol=1e-12, max_iter=100):
    """
    Optionally, demonstrate how thrust evolves over iteration for a single row.
    Similar to plot_thrust(...) in the original code.
    """
    thrust_values = []
    thrust = float(row['T_0'])
    for _ in range(max_iter):
        thrust_values.append(thrust)
        new_thrust = thrust_iteration(row, thrust)
        if abs(new_thrust - thrust) < tol:
            thrust_values.append(new_thrust)
            break
        thrust = new_thrust

    iterations = range(1, len(thrust_values) + 1)
    plt.figure()
    plt.plot(iterations, thrust_values, marker='o', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('Thrust Estimate')
    plt.title('Thrust Estimates Over Iterations (Single Row Example)')
    plt.grid(True)
    plt.show()

def plot_thrust_vs_alpha(df):
    """
    Filters for V == 40 and plots all Thrust vs AoA curves for different J values on one figure.
    """
    df_filtered = df[np.abs(df['V'] - 40) <= 1]
    unique_Js = df_filtered['J_M1'].unique()
    plt.figure()

    for j_val in unique_Js:
        subset = df_filtered[df_filtered['J_M1'] == j_val]
        subset = subset.sort_values('AoA')
        plt.plot(subset['AoA'], subset['thrust'], label=f'J = {j_val:.2f}')

    plt.xlabel('Angle of Attack (deg)')
    plt.ylabel('Thrust (N)')
    plt.title('Thrust vs. AoA for Different J Values (V = 40)')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Read the data
    file_path = 'uncorrected_elevator_-10.txt'
    df = read_data_to_df(file_path)
    df['delta_e'] = -10


    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_rows', None)  # Optional: Show all rows
    pd.set_option('display.width', 0)  # Prevent line wrapping
    print(df)

    # Run thrust correction for each row
    df = thrust_correction(df)

    # Plot thrust vs alpha
    plot_thrust_vs_alpha(df)
    # Optionally plot iteration details for one row (e.g. row 0)
    plot_thrust_iteration_example(df.loc[0])
    print(df.loc[0])

    # Export the updated DataFrame (with T_0 and final thrust) to a CSV
    df.to_csv('corrected_elevator_data.csv', index=False)