import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
import scipy
import time

def getEmanations(filename, start_time):
    num_samples = 32768
    num_bands = 3 # for 6GHz spectrum
    num_sweeps = 1

    freq_range = ["200m", "400m"]
    mag_orders = ['k','m','g']

    for i in range(len(mag_orders)):
        if mag_orders[i] in freq_range[0]:
            fLow = float(freq_range[0].split(mag_orders[i])[0])*10**(3*(i+1))
        if mag_orders[i] in freq_range[1]:
            fHi = float(freq_range[1].split(mag_orders[i])[0])*10**(3*(i+1))

    F_START = fLow
    F_S = 200e6
    F_HOP = 100e6
    N_HOPS = 2
    N_SWEEPS = 1

    data1 = np.fromfile(open(filename), dtype=scipy.complex64)

    fstart = F_START
    fhop = F_HOP
    n_hops = N_HOPS
    n_sweeps = N_SWEEPS
    fs = F_S

    n_samples_per_hop = len(data1)/(n_hops*n_sweeps)

    cut_start = np.floor(n_samples_per_hop*0.25)
    cut_stop = np.floor(n_samples_per_hop*0.75)-1
    cut_size = cut_stop - cut_start+1

    powers = []
    timestamps = []
    for i in range(1, n_sweeps+1):
        for j in range(1, n_hops+1):
            curr_start = int((j-1)*cut_size*2)
            curr_stop = int(j*cut_size*2)
            curr_data = data1[curr_start:curr_stop]
            sub_min_data = data1[curr_start:curr_stop] - np.mean(data1[curr_start:curr_stop])
            f, t, Sxx = signal.spectrogram(curr_data, nperseg=1024, noverlap=1024//2,mode='psd', return_onesided=False)
            pwelpsd = np.mean(Sxx.T, axis=0)
            power = 10*np.log10(pwelpsd)[np.argsort(f)]
            with open('GH_' + str(j*200) + 'CF_index.pkl', 'rb') as index_f:
                power_ix = np.load(index_f)
            powers.append(power[power_ix])
            timestamps.append(time.time() - start_time)
    return np.concatenate(powers), timestamps

#         peaks, _ = signal.find_peaks(pwelpsd, prominence=0.0000001)
#         fig, ax = plt.subplots()
#         ax.plot(f[np.argsort(f)], 10*np.log10(pwelpsd)[np.argsort(f)])
#         ax.plot(f[peaks], 10*np.log10(pwelpsd)[peaks], 'o')
#         plt.show()
