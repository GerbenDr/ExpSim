import csv
from dataclasses import dataclass
import numpy as np
from prop_off_RSM import rsm_CD_po, rsm_CM25c_po, rsm_CL_po
import matplotlib.pyplot as plt

@dataclass
class ElevatorData:
    run: int
    hr: int
    min: int
    sec: int
    AoA: float
    AoS: float
    dPb: float
    pBar: float
    temp: float
    B1: float
    B2: float
    B3: float
    B4: float
    B5: float
    B6: float
    rpmWT: float
    rho: float
    q: float
    V: float
    Re: float
    rpsM1: float
    rpsM2: float
    iM1: float
    iM2: float
    dPtQ: float
    tM1: float
    tM2: float
    vM1: float
    vM2: float
    pInf: float
    nu: float
    J_M1: float
    J_M2: float
    FX: float
    FY: float
    FZ: float
    MX: float
    MY: float
    MZ: float
    CFX: float
    CFY: float
    CFZ: float
    CMX: float
    CMY: float
    CMZ: float
    CN: float
    CT: float
    CY: float
    CL: float
    CD: float
    CYaw: float
    CMroll: float
    CMpitch: float
    CMpitch25c: float
    CMyaw: float
    v_i: float = 0.0
    T_0: float = 0.0
    thrust: float = 0.0  # Additional attribute for thrust

def read_elevator_data(file_path):
    data_list = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            # Convert all values to appropriate types
            row_data = {key: float(value) if '.' in value else int(value) for key, value in row.items()}
            # Create an instance of ElevatorData
            data = ElevatorData(**row_data)
            data_list.append(data)
    return data_list


# Example usage
file_path = 'uncorrected_elevator_10.txt'
elevator_data = read_elevator_data(file_path)

class dataprocessor:
    def __init__(self, data_list):
        self.data_list = data_list
        self.A_ref = 0.2
        self.A_rotor = 0.1
        self.airfoil_cd = 0.005

    def calculate_T0(self):
        for dp in self.data_list:
            predicted_CD_po = rsm_CD_po.predict([[dp.AoA, dp.V]])
            Fx_po = -predicted_CD_po * self.A_ref * dp.q
            Fx_pon = -dp.CD * self.A_ref * dp.q

            T_initial = Fx_pon - Fx_po
            dp.T_0 = T_initial

    def thrust_iteration(self, datapoint, thrust):
        v_i = 0.5*np.sqrt(datapoint.V + 2*thrust/(self.A_ref*datapoint.rho)) - datapoint.V/2
        D_po = 0.5*datapoint.rho*datapoint.V**2 * self.airfoil_cd * 2 * 0.2 * 0.1
        D_pon = 0.5*datapoint.rho*(datapoint.V + v_i)**2 * self.airfoil_cd * 2 * 0.2 * 0.1
        delta_D = D_po - D_pon
        thrust = float(datapoint.T_0) + delta_D
        return thrust

    def thrust_correction(self, dp, tol=1e-12, max_iter=100):
        thrust = float(dp.T_0)
        thrust_list = [thrust]
        for _ in range(max_iter):
            new_thrust = self.thrust_iteration(dp, thrust)
            if abs(new_thrust - thrust) < tol:
                break
            thrust = new_thrust
            thrust_list.append(thrust)
        return thrust_list, thrust


thrust_estimator = dataprocessor(elevator_data)
thrust_estimator.calculate_T0()
thrust_estimates = thrust_estimator.thrust_correction(thrust_estimator.data_list[0])[0]
print(thrust_estimates)

def plot_thrust(thrust_estimates):
    iterations = range(1, len(thrust_estimates) + 1)  # Iterations as integers starting from 1
    print(thrust_estimates)
    print(iterations)
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, thrust_estimates, marker='o', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('Thrust Estimate')
    plt.title('Thrust Estimates Over Iterations')
    plt.grid(True)
    plt.show()

# Example usage
plot_thrust(thrust_estimates)