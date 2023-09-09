import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
import os
from scipy.signal import find_peaks
from scipy.signal import medfilt

def getEmanations(filename):
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
    F_S = 250e6
    F_HOP = 100e6
    N_HOPS = 1
    N_SWEEPS = 50

    while not os.path.exists(filename):
        continue
    data1 = np.fromfile(filename, dtype=scipy.complex64)
    
    fstart = F_START
    fhop = F_HOP
    n_hops = N_HOPS
    n_sweeps = N_SWEEPS
    fs = F_S

    n_samples_per_hop = len(data1)/(n_hops*n_sweeps)

    cut_start = np.floor(n_samples_per_hop*0.25)
    cut_stop = np.floor(n_samples_per_hop*0.75)-1
    cut_size = cut_stop - cut_start+1

    power_result = np.zeros((n_sweeps, 5))
    for i in range(1, n_sweeps+1):
        for j in range(1, n_hops+1):
            n_start = int((i-1)*n_hops*n_samples_per_hop + (j-1)*n_samples_per_hop)+1
            n_stop = int((i-1)*n_hops*n_samples_per_hop + j*n_samples_per_hop)
            
            curr_data = data1[n_start:n_stop]

            #f, t, Sxx = signal.spectrogram(curr_data, nperseg=1024, noverlap=1024//2,mode='psd', return_onesided=False)
            #pwelpsd = np.mean(Sxx.T, axis=0)
            #power = 10*np.log10(pwelpsd)[np.argsort(f)]

            power = 20*np.log10(np.fft.fftshift(np.abs(np.fft.fft(curr_data))))

            power_good = power[int(cut_start):int(cut_stop)]
            # using move median to smooth the noise floor
            move_power = medfilt(power_good, 301)
            # find the peaks that are psd of the emanations
            final_power = power_good - move_power
            peaks, _ = find_peaks(final_power, height=1.4)
            
            power_ix = np.array([np.mean(final_power[peaks]), np.median(final_power[peaks]), np.std(final_power[peaks]), np.var(final_power[peaks]), np.average(final_power[peaks])])
            # power_result = np.vstack((power_result, power_ix))
            power_result[i - 1] = power_ix
            # curr_start = int((j-1)*cut_size)+1
            # curr_stop = int(j*cut_size)

            #print(power_ix)


            
            #powers.append(final_power[power_ix])

    return power_result        
    #return np.concatenate(powers)

#         peaks, _ = signal.find_peaks(pwelpsd, prominence=0.0000001)
#         fig, ax = plt.subplots()
#         ax.plot(f[np.argsort(f)], 10*np.log10(pwelpsd)[np.argsort(f)])
#         ax.plot(f[peaks], 10*np.log10(pwelpsd)[peaks], 'o')
#         plt.show()
