import numpy as np

# Aircraft Parameters
S_REF = 0.2172 # m^2 - reference area (=wing area)

# Fuselage Parameters
L_FUSELAGE = 1342e-3 # m
D_FUSELAGE = 0.140 # m
D_OVER_L_FUSELAGE = D_FUSELAGE / L_FUSELAGE # - fuselage fineness ratio
V_FUSELAGE = 0.0160632 # m^3
K3_FUSELAGE = 0.91 # - fuselage shape factor (based on the given fineness ratio)
TAU1_FUSELAGE = 0.865 # - tunnel/model factor, b=0 for body of revolution

# Aft Strut Parameters
V_AFTSTRUT = 0.0004491 # m^3
T_OVER_C_AFTSTRUT = 0.2 # - (t/c) ratio taken from section A-A of the drawing of the aft strut
K1_AFTSTRUT = 1.082 # - aft strut shape factor, according to NACA 4-digit airfoil and given t/c ratio
TAU1_AFTSTRUT  = 0.836 # - tunnel/model factor, based on strut being 0.46/0.89 = 0.516 of the tunnel height when it's attached to the centerline of the fuselage. Assuming B/H=1

# Wing Strut Parameters
V_WINGSTRUTS = 0.0035296 # m^3, both together
L_WINGSTRUTS = 0.61 # m
T_OVER_C_WINGSTRUTS = 5/19 # - (t/c) ratio taken from the CATPart, this is the thickes part of the wing strut, that which makes up most of the strut
# The wing strut has a t/c ratio of 18/40 at the smaller part of the strut.
# Modeling it as a NACA 4-digit airfoil
K1_WINGSTRUTS = 1.13 # - wing strut shape factor, according to NACA 4-digit airfoil and given t/c ratio
TAU1_WINGSTRUTS = 0.836 # - tunnel/model factor, based on strut being 0.46/0.89 = 0.516 of the tunnel height when it's attached to the centerline of the fuselage. Assuming B/H=1

# Wing Parameters
V_WING = 0.0030229 # m^3
B_WING = 1397e-3 # m
T_OVER_C_WING = 0.15 # - (t/c) ratio
K1_WING = 1.036 # - wing shape factor: analysis shows max t/c of airfoil at 27%, typically 30% for NACA 4-series, so that's best match (based on 4 digit NACA airfoil and given t/c ratio)
TAU1_WING = 0.885 # - tunnel/model factor, b/B 
B_WING_VORTEX = 0.74 * B_WING # m
B_WING_EFFECTIVE = (B_WING + B_WING_VORTEX) / 2 # m
C_WING = 0.165 # m - mean aerodynamic chord of the wing
DELTA = 0.103 # Boundary Correction factor. Based on elliptical wing, vortex span, effective span. Assumes wing is at centerline, not entirely true.
TAU2_HALFCHORD = 0.144 # - factor based on the given graph for elliptical tunnels as it was said earlier in the slides that elliptical correction mroe closely resembles octogonal section.
TAU2_TAILARM = 0.756 # - factor based on the given graph for elliptical tunnels as it was said earlier in the slides that elliptical correction mroe closely resembles octogonal section.

# Horziontal Tail Parameters
V_HTAIL_LESS_NACELLE = 0.0009751 # m^3
V_NACELLE = 0.0007921 # m^3 single nacelle volume
V_HTAIL_TOTAL = V_HTAIL_LESS_NACELLE + 2 * V_NACELLE # m^3
C_HTAIL = 0.9*0.165
B_HTAIL = 0.576 # m
T_OVER_C_HTAIL = 0.15 # - (t/c) ratio for NACA 64(2)-015A airfoil
K1_HTAIL = 1.019 # - horizontal tail shape factor (based on 64-series NACA airfoil and given t/c ratio)
TAU1_HTAIL = 0.863 # - tunnel/model factor
TAILARM = 0.535 # m quarter chord to quarter chord distance
TAIL_CHORD = 0.9 * C_WING # m

# Engine Nacelle Parameters
# Volume of nacelle is already included in the horizontal tail parameter section above
# Diameter of nacelle is calculated in derived parameter section
L_NACELLE = 0.345 # m
K3_NACELLE = 0.933 # - nacelle shape factor
TAU1_NACELLE = 0.865 # - tunnel/model factor

# Vertical Tail Parameters
V_VTAIL = 0.0003546 # m^3
T_OVER_C_VTAIL = 0.15 # - (t/c) ratio for NACA 0015 airfoil
B_VTAIL = 0.258 # m
K1_VTAIL = 1.036 # - vertical tail shape factor (based on 4 digit NACA airfoil and given t/c ratio)
TAU1_VTAIL = 0.814 # - tunnel/model factor: using the H/B of 1 as the best guess we have, but it is not accurate. Was suggested to be okay by Tomas in thread.

# Tunnel Parameters
B_TUNNEL = 1.29 * B_WING # m
H_TUNNEL = 0.89 * B_WING # m
B_OVER_H_TUNNEL = B_TUNNEL / H_TUNNEL
H_OVER_B_TUNNEL = 1 / B_OVER_H_TUNNEL
C_TUNNEL = B_TUNNEL * H_TUNNEL - 2 * 0.3**2 # m^2

# Propeller Parameters
D_PROP = 0.2032 # m
S_PROP = 0.25 * np.pi * D_PROP**2 # m^2 - propeller disk area

# Derived Parameters
BWING_OVER_B_TUNNEL = B_WING / B_TUNNEL # - 
BHTAIL_OVER_B_TUNNEL = B_HTAIL / B_TUNNEL # -
D_NACELLE = 0.28 * D_PROP # m
D_OVER_L_NACELLE = D_NACELLE / L_NACELLE # - nacelle fineness ratio
BVTAIL_OVER_H_TUNNEL = B_VTAIL / H_TUNNEL # -
K_WING = B_WING_EFFECTIVE / B_TUNNEL # - 
HALFCHORD_OVER_BTUNNEL = 0.5 * C_WING / B_TUNNEL # -
TAILARM_OVER_BTUNNEL = TAILARM / B_TUNNEL # -

# MASKING THE DATA
mask_RSM = np.hstack((
    np.arange(0, 8, 1),
    np.arange(10, 12, 1),
    np.arange(14, 19, 1),
    np.arange(21, 23, 1),
    np.arange(25, 30, 1),
    np.arange(31, 32, 1),
    # np.arange(33, 36, 1),
    np.arange(44, 53, 1),
    np.arange(53, 64, 1),
    np.arange(69, 74, 1),
    np.arange(79, 82, 1),
    np.arange(86, 87, 1),
    np.arange(91, 92, 1),
))

mask_validation = np.hstack((
    np.arange(32, 35, 1),
    np.arange(92, 95, 1),
))

mask_repetition_exclusive = np.hstack((
    np.arange(8, 10, 1),
    np.arange(12, 14, 1),
    np.arange(19, 21, 1),
    np.arange(23, 25, 1),
    np.arange(65, 69, 1),
    np.arange(75, 79, 1),
    np.arange(82, 86, 1),
    np.arange(88, 91, 1),
))

mask_repetition_inclusive = np.hstack((
    np.arange(7, 10, 1),
    np.arange(11, 14, 1),
    np.arange(18, 21, 1),
    np.arange(22, 25, 1),
    np.arange(63, 69, 1),
    np.arange(73, 79, 1),
    np.arange(81, 86, 1),
    np.arange(87, 91, 1),
))

mask_repetition_pointwise_inclusive = [
    np.arange(7, 10, 1),
    np.arange(11, 14, 1),
    np.arange(18, 21, 1),
    np.arange(22, 25, 1),
    np.arange(64, 69, 1),
    np.arange(74, 79, 1),
    np.arange(81, 86, 1),
    np.arange(87, 91, 1),
]

mask_low_Re = np.hstack((
    np.arange(35, 44, 1),
))

print(B_OVER_H_TUNNEL)