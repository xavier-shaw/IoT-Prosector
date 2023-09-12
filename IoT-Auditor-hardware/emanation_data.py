import psutil
import time
import subprocess
import FFTpeaks
import os 
import pickle

def emanation_data(idx):
    debug = False
    if debug:
        return 1
    else:
        absolute_path = "/home/datasmith/Desktop/Iot-Auditor/IoT-Auditor/IoT-Auditor-hardware/"
        relative_path_1 = "getEmanations_2.sh"
        relative_path_2 = "em_data/em_" + idx + ".32cf"
        full_path_1 = os.path.join(absolute_path, relative_path_1)
        full_path_2 = os.path.join(absolute_path, relative_path_2)
        subprocess.run([full_path_1, idx], capture_output=True, text=True)
        ## rayray's emanation detection codes translated from matlab
        fft_peaks =  FFTpeaks.getEmanations_raw(full_path_2)
        # q.put(fft_peaks)
        file_name = "/home/datasmith/Desktop/Iot-Auditor/IoT-Auditor/IoT-Auditor-hardware/fft_result/" + idx + ".pkl"
        with open(file_name, 'wb') as file:
            pickle.dump(fft_peaks, file)
        return True
    
def recalculate_emanation_data(idx):
    debug = False
    if debug:
        return 1
    else:
        absolute_path = "/home/datasmith/Desktop/Iot-Auditor/IoT-Auditor/IoT-Auditor-hardware/"
        # relative_path_1 = "getEmanations_2.sh"
        relative_path_2 = "em_data/em_" + idx + ".32cf"
        # full_path_1 = os.path.join(absolute_path, relative_path_1)
        full_path_2 = os.path.join(absolute_path, relative_path_2)
        # subprocess.run([full_path_1, idx], capture_output=True, text=True)
        ## rayray's emanation detection codes translated from matlab
        fft_peaks =  FFTpeaks.getEmanations_raw(full_path_2)
        return fft_peaks