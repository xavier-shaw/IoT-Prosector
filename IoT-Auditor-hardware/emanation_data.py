import psutil
import time
import subprocess
import FFTpeaks
import os 

def emanation_data(q):
    debug = False
    if debug:
        return 1
    else:
        absolute_path = os.path.dirname(__file__)
        relative_path_1 = "getEmanations_2.sh"
        relative_path_2 = "gh_data/gh.32cf"
        full_path_1 = os.path.join(absolute_path, relative_path_1)
        full_path_2 = os.path.join(absolute_path, relative_path_2)
        subprocess.call(full_path_1, shell=True)
        ## rayray's emanation detection codes translated from matlab
        fft_peaks =  FFTpeaks.getEmanations(full_path_2)
        q.put(fft_peaks)
        return(fft_peaks)