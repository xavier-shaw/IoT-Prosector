import serial
def power_data(q):
    power = []
    with serial.Serial('/dev/ttyACM0', 9600, timeout=1) as ser:
        for i in range(8):
            line = ser.readline()
            if len(line.decode().rstrip().split(',')[-1])>=1:
               current = float(line.decode().rstrip().split(',')[-1])
               if current<1:
                   power.append(current)
            #print(i)
    q.put(power)
    return(power)
    """
    filename = 'file.csv'
    with open(filename, 'w') as f:
        f.write(''.join(power))
    """

