import psutil
import time
import subprocess
def network_data(q):
    debug = False
    if debug:
        return 1
    else:
        subprocess.call('./getPackets.sh', shell=True)
        lines = []
        with open('AIY5SecThroughput.txt','r') as f:
            for i, line in enumerate(f):
                if i > 11:
                    data = line.split('|')
                    if len(data) > 2:
                        lines.append(int(data[2].strip()))
                    else:
                        lines.append(0)
        q.put(lines)
        return(lines)

def network_data_g():
    debug = False
    if debug:
        return 1
    else:
        subprocess.call('./getPackets.sh', shell=True)
        lines = []
        with open('AIY5SecThroughput.txt','r') as f:
            for i, line in enumerate(f):
                if i > 11:
                    data = line.split('|')
                    if len(data) > 2:
                        lines.append(int(data[2].strip()))
                    else:
                        lines.append(0)
        return(lines)
