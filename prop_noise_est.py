import numpy as np

LTT_noise_floor_40 = 78 # dB at 40m/s, source: i made it up
LTT_noise_floor_20 = 71 # dB at 20m/s, source: i made it up

required_SNR = 6  # db
sample_rate = 51200 # Hz
window_width = 0.1  # s, completely arbitrary, gives us 10Hz of spectral resolution


# prop noise est, source: Ruijgrok
def prop_noise(CP, n, D, Vinf, r, B=6, N=2, rho = 1.25, a = 330):
    Pbr = CP * n**3 * D**5 * rho / 1000 # kW
    Mt = np.sqrt(Vinf**2 + (D * np.pi * n)**2) / a

    return 83.4 + 15.3 * np.log10(Pbr) - 20 * np.log10(D) + 38.5 * Mt - 3 * (B-2) + 10 * np.log10(N) - 20 * np.log10(r)

def CP(J):
    return -0.0093 * J**4 + 0.1832 * J**3 - 1.1784 * J**2 + 2.2005 * J - 0.5180

if __name__ == "__main__":
    Jmax = 2.4
    D = 0.205
    r_mic = np.array([55, 44, 43]) / 100
    r = np.linalg.norm(r_mic)
    for V, NFzero in zip([20, 40], [LTT_noise_floor_20, LTT_noise_floor_40]):
        n = V / Jmax / D
        NPmin = prop_noise(CP(Jmax), n, D, V, r)

        NF = NFzero

        SNR = NPmin - NF

        delta_SNR = required_SNR - SNR

        n_samples_required = 10**(delta_SNR/10)
        measurement_time_required = window_width * n_samples_required

        print('tunnel speed: {} m/s'.format(V))
        print('prop noise: {} dB'.format(NPmin))
        print('tunnel noise: {} dB'.format(NF))
        print('SNR: {} dB'.format(SNR))
        print('samples required for SNR = {} dB: {}'.format(required_SNR, n_samples_required))
        print('minimum sample time: {} s'.format(measurement_time_required))




