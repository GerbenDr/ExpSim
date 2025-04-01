## Methods to implement boundary corrections

# Import necessary packages
import numpy as np
import pandas as pd
import constants as c
import matplotlib.pyplot as plt

## Obtain the drag polar from a dataframe
def get_drag_polar(df, visualise = False, tail_on=True):
    """
    Obtain the drag polar from the given dataframe.
    This function calculates the drag polar from the given dataframe by fitting a 
    second order polynomial to the data. The coefficients of the polynomial are then 
    returned as a tuple (CD0, a, b) where CD0 is the zero-lift drag coefficient, a is 
    the induced drag coefficient, and b is the parasite drag coefficient. The drag polar 
    is assumed to be of the form CD = CD0 + a*CL^2.
    Parameters:
    df (pandas.DataFrame): The input dataframe containing the measured data.
    Returns:
    tuple: A tuple containing the zero-lift drag coefficient, the induced drag coefficient, 
           and the parasite drag coefficient.
    """
    
    # If visualise, plot the drag polar points to which the polynomial will be fitted
    if False:
        plt.scatter(df['CL'], df['CD_thrust_cor'])
        plt.xlabel('CL')
        plt.ylabel('CD')
        plt.title('Measured Data')
        plt.grid()
        plt.show()
        
    # Fit a second order polynomial to the data
    if tail_on:
        p = np.polyfit(df['CL']**2, df['CD_thrust_cor'], 1)
    else:
        df = df[(df['CL']<0.95)]
        p = np.polyfit(df['CL']**2, df['CD'], 1)
    
    # Extract the coefficients
    a_CDi = p[0]
    CD0 = p[1]
    
    # Visualise the drag polar if requested
    if visualise:
        # Create a range of CL values
        CL_range = np.linspace(0, 1.2, 100)
        
        # Calculate the corresponding CD values
        CD_range = CD0 + a_CDi * CL_range**2
        
        # Plot the drag polar
        plt.figure()
        plt.plot(CL_range, CD_range, label = 'Drag Polar Fit')
        if tail_on:
            plt.scatter(df['CL'], df['CD_thrust_cor'], label = 'Measured Data')
        else:
            plt.scatter(df['CL'], df['CD'], label = 'Measured Data')
        plt.xlabel('CL')
        plt.ylabel('CD')
        plt.title('Drag Polar')
        plt.legend()
        plt.grid()
        plt.show()
    
    
    return CD0, a_CDi

## Obtain CL-alpha curve for the aircraft-less-tail configuration
# This curve is used to calculate the wing lift coefficient for the lift-interference correction
def get_CL_alpha_curve_less_tail(df, V, visualise = False):
    
    # Filter the data to only include the measurements at V m/s +/- 1%
    df_V = df[(df['V'] >= 0.99*V) & (df['V'] <= 1.01*V)]
    
    # Fit a third order polynomial to the data
    p = np.polyfit(df_V['AoA'], df_V['CL_cor1'], 3)
    
    # Extract the coefficients
    a = p[0]
    b = p[1]
    c = p[2]
    d = p[3]
    
    # Visualise the CL-alpha curve if requested
    if visualise:
        # Create a range of AoA values
        AoA_range = np.linspace(-10, 10, 100)
        
        # Calculate the corresponding CL values
        CL_range = a * AoA_range**3 + b * AoA_range**2 + c * AoA_range + d
        
        # Plot the CL-alpha curve
        plt.figure()
        plt.plot(AoA_range, CL_range, label = 'CL-alpha Curve Fit')
        plt.scatter(df_V['AoA'], df_V['CL_cor2'], label = 'Measured Data')
        plt.xlabel('AoA')
        plt.ylabel('CL')
        plt.title('CL-alpha Curve')
        plt.legend()
        plt.grid()
        plt.show()
    
    return a, b, c, d

## Obtain CM-alpha curve for the tail: change in aircraft moment coefficient with tail angle of attack.
def get_CM_alpha_curve_tail():
    
    a0 = 2*np.pi # This is a guess, but it is a good guess. It is the lift curve slope of the tail.

    AR_TAIL = c.B_HTAIL / c.TAIL_CHORD # Aspect ratio of the tail

    a_3d_corrected = a0 / (1+a0 / np.pi / AR_TAIL)

    CL_alpha_tail = a_3d_corrected * np.pi /180

    CM_alpha_tail = -CL_alpha_tail * c.TAILARM / c.C_WING # This is the moment curve slope of the tail.    
    
    return CM_alpha_tail

## Subtract the model-off data from the uncorrected dataset
def subtract_model_off_data(df_unc, df_model_off, tail_on=True):
    """
    Subtract the model-off data from the uncorrected dataset.
    This function subtracts the model-off data from the uncorrected dataset to account for 
    the support loads. The corrected dataset is then returned.
    Parameters:
    df_unc (pandas.DataFrame): The uncorrected dataset.
    df_model_off (pandas.DataFrame): The model-off dataset.
    Returns:
    pandas.DataFrame: The corrected dataset.
    """
    
    # The df_model_off data only has data for a set of anles of attack.
    # It needs to be interpolated for each row in df_unc to apply the correction on CL, CD, Cy, CMpitch, CMpitch, and CMyaw.
    # For each row in df_unc, find the two closest angles of attack in df_model_off.
    # Then, interpolate the model-off data (CL, CD, Cy, CMpitch, CMpitch, and CMyaw) linearly between these two angles of attack.
    # Finally, subtract this interpolated model-off data from the uncorrected data.
    
    # Create an empty dataframe to store the model-off data interpolated for each row in df_unc
    df_model_off_interpolated = pd.DataFrame(columns = df_model_off.columns)
    
    # Iterate over each row in df_unc
    for index, row in df_unc.iterrows():
                
        # Find the two closest angles of attack in df_model_off
        upper = df_model_off[df_model_off['AoA'] >= row['AoA']].iloc[0]
        lower = df_model_off[df_model_off['AoA'] < row['AoA']].iloc[-1] if not df_model_off[df_model_off['AoA'] < row['AoA']].empty else df_model_off.iloc[0]
                
        # Interpolate the model-off data linearly between these two angles of attack
        # Calculate the interpolation factor
        factor = (row['AoA'] - lower['AoA']) / (upper['AoA'] - lower['AoA'])
                
        # Interpolate the model-off data
        interpolated = lower + factor * (upper - lower)
        
        # Append the interpolated data to the dataframe
        df_model_off_interpolated = pd.concat([df_model_off_interpolated, interpolated.to_frame().T], ignore_index=True)
            
    # # Reset the df_unc index
    
    df_unc = df_unc.reset_index(drop=True)

    # Subtract the interpolated model-off data from the uncorrected data for each component
    df_unc['CL'] = df_unc['CL'] - df_model_off_interpolated['CL']
    if tail_on:
        df_unc['CD_thrust_cor'] = df_unc['CD_thrust_cor'] - df_model_off_interpolated['CD']
    else:
        df_unc['CD'] = df_unc['CD'] - df_model_off_interpolated['CD']
    df_unc['CY'] = df_unc['CY'] - df_model_off_interpolated['Cy']
    df_unc['CMroll'] = df_unc['CMroll'] - df_model_off_interpolated['CMroll']
    df_unc['CMpitch'] = df_unc['CMpitch'] - df_model_off_interpolated['CMpitch']
    df_unc['CMyaw'] = df_unc['CMyaw'] - df_model_off_interpolated['CMyaw']
            
    return df_unc
    
## Method to apply the solid blockage corrections
def calculate_solid_blockage_corrections(df, tail_on=True):
    """
    Apply solid blockage corrections to the given dataframe.
    This function calculates the solid blockage corrections for different 
    components of an aircraft (fuselage, wing, horizontal tail, engine nacelles, 
    and vertical tail) based on constants and formulas provided. The total solid 
    blockage correction is then added as a new column 'epsilon_sb' to the dataframe.
    Parameters:
    df (pandas.DataFrame): The input dataframe to which the solid blockage corrections 
                           will be applied.
    tail_on (bool): A flag indicating whether the horizontal tail and engine nacelles 
                    are included in the correction.
    Returns:
    pandas.DataFrame: The dataframe with the added 'epsilon_sb' column containing the 
                      total solid blockage correction.
    """
    
    # Fuselage
    epsilon_sbf = c.K3_FUSELAGE * c.TAU1_FUSELAGE * c.V_FUSELAGE / (c.C_TUNNEL**(3/2))

    # Wing
    epsilon_sbw = c.K1_WING * c.TAU1_WING * c.V_WING / (c.C_TUNNEL**(3/2))
    
    # Vertical Tail
    epsilon_sbvt = c.K1_VTAIL * c.TAU1_VTAIL * c.V_VTAIL / (c.C_TUNNEL**(3/2))
    
    # Rear Strut
    epsilon_sbrs = c.K1_AFTSTRUT * c.TAU1_AFTSTRUT * c.V_AFTSTRUT / (c.C_TUNNEL**(3/2))
    
    # Wing Struts
    epsilon_sbws = 2 * (c.K1_WINGSTRUTS * c.TAU1_WINGSTRUTS * c.V_WINGSTRUTS / (c.C_TUNNEL**(3/2)))

    if tail_on:
        # Horizontal Tail
        epsilon_sbht = c.K1_HTAIL * c.TAU1_HTAIL * c.V_HTAIL_LESS_NACELLE / (c.C_TUNNEL**(3/2))

        # Engine Nacelles
        epsilon_sbn = 2 * (c.K3_NACELLE * c.TAU1_NACELLE * c.V_NACELLE / (c.C_TUNNEL**(3/2)))

    else:
        # Horizontal Tail
        epsilon_sbht = 0

        # Engine Nacelles
        epsilon_sbn = 0

    # Total Solid Blockage
    epsilon_sb = epsilon_sbf + epsilon_sbw + epsilon_sbht + epsilon_sbn + epsilon_sbvt + epsilon_sbrs + epsilon_sbws

    # Add the epsilon_sb column to the dataframe
    df['epsilon_sb'] = epsilon_sb

    return df

## Method to apply the wake blockage corrections
def calculate_wake_blockage_corrections(df, CD0, a_CDi, tail_on=True):
    """
    Apply wake blockage corrections to the given dataframe.
    This function calculates and applies wake blockage corrections based on the provided
    zero-lift drag (CD0) and induced drag (CDi) coefficients. The corrections are based on
    formulas from slide 114 of lecture 3/4. Stall wake blockage is initially neglected as
    the assumption is we stayed outside of the stall regime.
    Parameters:
    df (pandas.DataFrame): The dataframe containing the measured data.
    CD0 (float): The zero-lift drag coefficient.
    CDi (float): The induced drag coefficient.
    Returns:
    pandas.DataFrame: The dataframe with the added 'epsilon_wb' column containing the 
                      total wake-blockage correction.
    """

    # Zero-lift wake blockage
    epsilon_wb_0 = c.S_REF / 4 / c.C_TUNNEL * CD0

    # Stall wake blockage
    if tail_on:
        epsilon_wb_s = 5 * c.S_REF / 4 / c.C_TUNNEL * (df['CD_thrust_cor'] - CD0 - a_CDi * df['CL']**2)
    else:
        epsilon_wb_s = 5 * c.S_REF / 4 / c.C_TUNNEL * (df['CD'] - CD0 - a_CDi * df['CL']**2)
        
    # Total wake blockage
    epsilon_wb = epsilon_wb_0 + epsilon_wb_s
    
    # Add the epsilon_wb column to the dataframe
    df['epsilon_wb'] = epsilon_wb
    
    return df

## Method to apply the slipstream blockage corrections
def calculate_slipstream_blockage_corrections(df, tail_on=True):
    """
    Apply slipstream blockage corrections to the given dataframe.
    The corrections are based on formulas from slide 122 of lecture 3/4.
    It requires the thrust to be known.
    Parameters:
    df (pandas.DataFrame): The dataframe containing the measured data.
    Returns:
    pandas.DataFrame: The dataframe with the added 'epsilon_ss' column containing the 
                        slipstream-blockage correction.
    """
    
    if tail_on:
        # Calculate the thrust coefficient
        T_C = df['thrust'] / (df['rho'] * df['V']**2 * c.S_PROP) # Thrust coefficient

        # Slipstream blockage
        epsilon_ss = - T_C / 2 / np.sqrt(1 + 2 * T_C) * c.S_PROP / c.C_TUNNEL
        
        # Add the epsilon_ss column to the dataframe
        df['epsilon_ss'] = epsilon_ss
    else:
        # Slipstream blockage
        epsilon_ss = 0
        
        # Add the epsilon_ss column to the dataframe
        df['epsilon_ss'] = epsilon_ss
    
    return df
    
## Method to apply the total blockage corrections
def apply_total_blockage_corrections(df, tail_on=True, visualise=False):
    """
    Apply total blockage corrections to the given dataframe.
    This function calculates the total blockage corrections by summing the solid, wake, and 
    slipstream blockage corrections. The total blockage correction is then added as a new 
    column 'epsilon_total' to the dataframe. 
    
    Additionally, the corrected speed V_coris
    calculated by multiplying the measured speed V by the total blockage correction
    (V_cor= V * (1 + epsilon_total)). Equally, the corrected dynamic pressure q_coris
    calculated by multiplying the measured dynamic pressure q by the total blockage correction
    squared (q_cor= q * (1 + epsilon_total)^2).
    
    Additionally, the corrected lift-, drag-, and moment coefficients are calculated by
    multiplying the measured coefficients by the inverse of the total blockage correction squared. 
    
    Parameters:
    df (pandas.DataFrame): The input dataframe to which the total blockage corrections 
                           will be applied.
    Returns:
    pandas.DataFrame: The dataframe with the added 'epsilon_total', 'V_corr', and 'q_corr'
                      columns containing the total blockage correction, corrected speed, and
                      corrected dynamic pressure, respectively.
    """
        
    # Calculate solid blockage correction
    df = calculate_solid_blockage_corrections(df, tail_on)
    
    # Calculate wake blockage correction: this is done for delta_e = 10 and -10 separately
    # TODO: Review selection of data -> drag coefficients for negative lift coefficients are diverging, so they have been left out for now (CL>0)
    if tail_on:
        df_plus10 = df[(df['delta_e'] == 10) & (df['V'] > 35)]
        df_min10 = df[(df['delta_e'] == -10) & (df['V'] > 35)]
        df_V20 = df[(df['V'] > 18) & (df['V'] < 22)]
    
        # Apply the wake blockage correction to the delta = +10 data
        CD0, a_CDi = get_drag_polar(df_plus10[(df['CL'] > 0)], visualise = visualise)
        df_plus10 = calculate_wake_blockage_corrections(df_plus10, CD0, a_CDi)
        
        # Apply the wake blockage correction to the delta = -10 data
        CD0, a_CDi = get_drag_polar(df_min10[(df['CL'] > 0)], visualise = visualise)
        df_min10 = calculate_wake_blockage_corrections(df_min10, CD0, a_CDi)
        
        # Apply the wake blockage correction to the V = 20 data
        CD0, a_CDi = get_drag_polar(df_V20[(df['CL'] > 0)], visualise = visualise)
        df_V20 = calculate_wake_blockage_corrections(df_V20, CD0, a_CDi)
        
        # Combine the dataframes
        df = pd.concat([df_plus10, df_min10, df_V20])
    else:
        df = df[(df['V'] > 35)]
        CD0, a_CDi = get_drag_polar(df, visualise = visualise, tail_on=False)
        df = calculate_wake_blockage_corrections(df, CD0, a_CDi, tail_on=False)
    
    # Calculate slipstream blockage correction
    df = calculate_slipstream_blockage_corrections(df, tail_on)
    
    # Total blockage correction
    epsilon_total = df['epsilon_sb'] + df['epsilon_wb'] + df['epsilon_ss']
    
    # Corrected speed
    V_cor = df['V'] * (1 + epsilon_total)
    
    # Corrected dynamic pressure
    q_cor = df['q'] * (1 + epsilon_total)**2
    
    # Corrected lift coefficient
    CL_cor1 = df['CL'] / (1 + epsilon_total)**2
    
    # Corrected drag coefficient
    if tail_on:
        CD_cor1 = df['CD_thrust_cor'] / (1 + epsilon_total)**2
    else:
        CD_cor1 = df['CD'] / (1 + epsilon_total)**2
    
    # Corrected moment coefficient
    # TODO: check the correct CM column name i.e. quarter-chord or not
    CM_cor1 = df['CMpitch'] / (1 + epsilon_total)**2
    
    # Add the epsilon_total, V_corr, and q_corcolumns to the dataframe
    df['epsilon_total'] = epsilon_total
    df['V_cor'] = V_cor
    df['q_cor'] = q_cor
    df['CL_cor1'] = CL_cor1
    df['CD_cor1'] = CD_cor1
    df['CMpitch_cor1'] = CM_cor1
    
    return df

## Method to apply the lift-interference correction
def apply_lift_interference_correction(df, df_less_tail):
    
    # Calculate the AoA correction due to upwash
    # The CL-alpha curve is calculated based on the corrected tail-less data.
    CL_alpha_wing_poly = get_CL_alpha_curve_less_tail(df_less_tail, V=40, visualise = False)
    a, b, c_coeff, d = CL_alpha_wing_poly
    CL_WING = a * df['AoA']**3 + b * df['AoA']**2 + c_coeff * df['AoA'] + d
    delta_alpha_upwash = c.DELTA * c.S_REF / c.C_TUNNEL * CL_WING / 2 / np.pi * 360
    
    # Calculate the AoA correction due to upwash gradient
    delta_alpha_upwash_gradient = c.TAU2_HALFCHORD * delta_alpha_upwash
    
    # Calculate the total AoA correction
    delta_alpha_total = delta_alpha_upwash + delta_alpha_upwash_gradient
    
    # Calculate the corrected angle of attack
    alpha_cor = df['AoA'] + delta_alpha_total
    
    # Calculate the drag coefficient correction due to the change in AoA
    delta_CDw = c.DELTA * c.S_REF / c.C_TUNNEL * CL_WING**2
    
    # Calculate the corrected drag coefficient
    CD_cor2 = df['CD_cor1'] + delta_CDw
    
    # Calculate the moment coefficient correction due to upwash
    # NOTE: CL_alpha of the wing is assumed to be needed here, not aircraft CL-Alpha.
    delta_CM_upwash = 1 / 8 * delta_alpha_upwash_gradient * CL_alpha_wing_poly[2]
    
    # Calculate the tailplane angle of attack correction
    delta_alpha_tail = c.DELTA * c.S_REF / c.C_TUNNEL * CL_WING * (1 + c.TAU2_TAILARM) / 2 / np.pi * 360
    
    # Calculate the moment coefficient correction due to the tail
    CM_alpha_tail = get_CM_alpha_curve_tail()
    delta_CM_tail = delta_alpha_tail * CM_alpha_tail
    
    # Calculate the total moment coefficient correction
    delta_CM_total = delta_CM_upwash + delta_CM_tail
    
    # Calculate the corrected moment coefficient
    CM_cor2 = df['CMpitch_cor1'] + delta_CM_total
    
    # Add the delta_alpha_total, delta_CDw, and AoA_corcolumns to the dataframe
    df['delta_alpha_total'] = delta_alpha_total
    df['delta_CL'] = 0 # No correction for lift coefficient
    df['delta_CDw'] = delta_CDw
    df['delta_CMpitch_total'] = delta_CM_total
    df['AoA_cor'] = alpha_cor
    df['CL_cor2'] = df['CL_cor1'] # No correction for lift coefficient
    df['CD_cor2'] = CD_cor2
    df['CMpitch_cor2'] = CM_cor2
    
    return df    

## Method to apply the lift-interference correction for tail-off configuration
def apply_lift_interference_correction_less_tail(df_less_tail):
    
    # Calculate the AoA correction due to upwash
    CL_alpha_wing_poly = get_CL_alpha_curve_less_tail(df_less_tail, V=40, visualise = False)
    a, b, c_coefficient, d = CL_alpha_wing_poly
    CL_WING = a * df_less_tail['AoA']**3 + b * df_less_tail['AoA']**2 + c_coefficient * df_less_tail['AoA'] + d
    delta_alpha_upwash = c.DELTA * c.S_REF / c.C_TUNNEL * CL_WING / 2 / np.pi * 360
    
    # Calculate the AoA correction due to upwash gradient
    delta_alpha_upwash_gradient = c.TAU2_HALFCHORD * delta_alpha_upwash
    
    # Calculate the total AoA correction
    delta_alpha_total = delta_alpha_upwash + delta_alpha_upwash_gradient
    
    # Calculate the corrected angle of attack
    alpha_cor = df_less_tail['AoA'] + delta_alpha_total
    
    # Calculate the drag coefficient correction due to the change in AoA
    delta_CDw = c.DELTA * c.S_REF / c.C_TUNNEL * CL_WING**2
    
    # Calculate the corrected drag coefficient
    CD_cor2 = df_less_tail['CD_cor1'] + delta_CDw
    
    # Calculate the moment coefficient correction due to upwash
    delta_CM_upwash = 1 / 8 * delta_alpha_upwash_gradient * CL_alpha_wing_poly[2]
    
    # Calculate the total moment coefficient correction
    delta_CM_total = delta_CM_upwash
    
    # Calculate the corrected moment coefficient
    CM_cor2 = df_less_tail['CMpitch_cor1'] + delta_CM_total
    
    # Add the delta_alpha_total, delta_CDw, and AoA_corcolumns to the dataframe
    df_less_tail['delta_alpha_total'] = delta_alpha_total
    df_less_tail['delta_CL'] = 0 # No correction for lift coefficient
    df_less_tail['delta_CDw'] = delta_CDw
    df_less_tail['delta_CMpitch_total'] = delta_CM_total
    df_less_tail['AoA_cor'] = alpha_cor
    df_less_tail['CL_cor2'] = df_less_tail['CL_cor1'] # No correction for lift coefficient
    df_less_tail['CD_cor2'] = CD_cor2
    df_less_tail['CMpitch_cor2'] = CM_cor2
    
    return df_less_tail    