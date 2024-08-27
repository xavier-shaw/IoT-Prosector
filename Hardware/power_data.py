import serial
import time

def power_data(q, start_time):
    power = []
    timestamps = []
    
    ser = serial.Serial('/dev/tty.usbmodem21101', 9600, timeout=1)

    for i in range(8):
        print("time 2: ", time.time() - start_time)
        line = ser.readline()
        print("line: ", line)
        if line:
            current = float(line.decode().rstrip())
        #    if current<1:
            power.append(current)
            timestamps.append(time.time() - start_time)
        #print(i)
    
    q.put(power)
    q.put(timestamps)
    
    return(power, timestamps)
    """
    filename = 'file.csv'
    with open(filename, 'w') as f:
        f.write(''.join(power))
    """

