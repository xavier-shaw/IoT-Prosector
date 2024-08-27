import tkinter as tk
import network_data
import power_data
import emanation_data
import numpy as np
import time
from scipy import stats
import pickle
from sklearn.cluster import KMeans
import multiprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


canvas_width = 1000
canvas_height = 1000

window = tk.Tk()
#window.geometry("800x1000")
window.title("IoTAuditor")
canvas = tk.Canvas(window, width=canvas_width, height=canvas_height)
canvas.pack()


#  for debugging purpose
debug = False

# box parameters
x0 = 500 # original x axis for the first box
y0 = 50 # original y axis for the first box
shift = 10 # backward arrow 

width = 260 # width of the box
height = 80 # heigh of the box
dis_box = 60 # distance between two boxes
#states = ['power_on_unmuted', 'power_on_muted', 'change_volume_muted', 'interact_muted', 'interact_unmuted', 'change_volume_unmuted']
global prev_box, power_on_unmuted_box, power_on_muted_box, change_volume_muted_box, interact_muted_box, interact_unmuted_box, change_volume_unmuted_box
prev_box = None
internet_box = None
listen_box = None
speech_process_box = None
response_box = None



# create root box, which is power-off state
root_state = canvas.create_rectangle(x0,y0, x0+width, y0+height, fill='white')
text_xp = x0 + width/2
text_yp = y0 + height/2
root_box = canvas.create_text(text_xp, text_yp, text="Power off", font=("Times New Roman", 20))
canvas.update()

# get the std of last state
std_pre = 0
# count the number of boxes
num = 1

# position of the last box
x_pre = x0
y_pre = y0

pre_network = 0
while(num<7):
    num = num + 1
    if True:
       status = num-1
       time.sleep(5)
    else:
        while(True):
           # network traffic data capture and processing
           network = network_data.network_data_g()
           network_sum = np.sum(network)
           if network_sum>10 or (num==5 and pre_network>10):
               pre_network = network_sum
               status = num-1
               #print(network_mean)
               break
           else:
               continue
    # create a new box
    if prev_box is not None: 
        canvas.itemconfig(prev_box, fill='white')
    if num==2: # power on unmuted
        x_p = x_pre - width/2
        y_p = y_pre + dis_box*2
        # backward arrow
        start_x = x_p + width/2
        start_y = y_p
        end_x = x_pre + width/2
        end_y = y_pre + height

        forward_event = canvas.create_line(start_x, start_y, end_x, end_y, arrow=tk.LAST)
        # arrow positions
        power_off_to_power_on_unmuted_x1 = start_x
        power_off_to_power_on_unmuted_y1 = start_y
        power_off_to_power_on_unmuted_x2 = end_x
        power_off_to_power_on_unmuted_y2 = end_y

        backward_event = canvas.create_line(end_x+shift, end_y, start_x+shift, start_y, arrow=tk.LAST)
        # arrow positions
        power_on_unmuted_to_power_off_x1 = end_x+shift
        power_on_unmuted_to_power_off_y1 = end_y

        power_on_unmuted_to_power_off_x2 = start_x+shift
        power_on_unmuted_to_power_off_y2 = start_y


        power_on_unmuted_x = start_x
        power_on_unmuted_y = start_y

    elif num==3: # power on muted
        x_p = x_pre + width/2
        y_p = y_pre + dis_box*2
        forward_event = canvas.create_line(start_x, start_y, end_x, end_y, arrow=tk.LAST)

        # backward arrow
        start_x = x_p + width/2
        start_y = y_p
        end_x = x_pre + width/2
        end_y = y_pre + height
        backward_event = canvas.create_line(end_x+shift, end_y, start_x+shift, start_y, arrow=tk.LAST)
        power_on_muted_to_power_on_unmuted_x1 = end_x+shift
        power_on_muted_to_power_on_unmuted_y1 = end_y
        power_on_muted_to_power_on_unmuted_x2 = start_x + shift
        power_on_muted_to_power_on_unmuted_y2 = start_y

        # power on muted -> power off
        mid_x = start_x + 50
        mid_y = (start_y + end_y)/2 

        end_power_x_off = x0 + width/2
        end_power_y_off = y0 + height
        power_on_muted_off = canvas.create_line(start_x, start_y, mid_x, mid_y, end_power_x_off, end_power_y_off, arrow=tk.LAST)
        
        power_on_muted_to_power_off_x = mid_x
        power_on_muted_to_power_off_y = mid_y

        power_on_muted_x = start_x
        power_on_muted_y = start_y


    elif num==4: # change volume on muted device
        x_p = x_pre - width/3
        y_p = y_pre + dis_box*2
        forward_event = canvas.create_line(start_x, start_y, end_x, end_y, arrow=tk.LAST)
        # backward arrow
        start_x = x_p + width/2
        start_y = y_p
        end_x = x_pre + width/2
        end_y = y_pre + height
        backward_event = canvas.create_line(end_x+shift, end_y, start_x+shift, start_y, arrow=tk.LAST)

        change_volume_muted_to_power_on_muted_x1 = end_x+shift
        change_volume_muted_to_power_on_muted_y1 = end_y
        change_volume_muted_to_power_on_muted_x2 = start_x+shift
        change_volume_muted_to_power_on_muted_y2 = start_y



        # change volume on muted device -> power off
        mid_temp_x1 = start_x + 300 
        mid_temp_y1 = start_y - 100
        change_volume_muted_off = canvas.create_line(start_x + width/2, start_y + height/2, mid_temp_x1, mid_temp_y1, end_power_x_off, end_power_y_off, arrow=tk.LAST)

        change_volume_muted_to_power_off_x = mid_temp_x1
        change_volume_muted_to_power_off_y = mid_temp_y1

        change_volume_on_muted_x = start_x
        change_volume_on_muted_y = start_y

    elif num==5: # interact with muted device
        x_p = x_pre + width/2
        y_p = y_pre + dis_box*2
        forward_event = canvas.create_line(start_x, start_y, end_x, end_y, arrow=tk.LAST)

        # backward arrow
        start_x = x_p + width/2
        start_y = y_p
        end_x = x_pre + width/2
        end_y = y_pre + height
        backward_event = canvas.create_line(end_x+shift, end_y, start_x+shift, start_y, arrow=tk.LAST)
        interaction_muted_to_change_volume_muted_x1 = end_x+shift
        interaction_muted_to_change_volume_muted_y1 = end_y
        interaction_muted_to_change_volume_muted_x2 =  start_x+shift
        interaction_muted_to_change_volume_muted_y2 =  start_y


        # interact with muted device -> power off
        mid_temp_x2 = start_x + 320
        mid_temp_y2 = start_y - 100
        interact_muted_off = canvas.create_line(start_x+width/2, start_y + height/2, mid_temp_x2, mid_temp_y2, end_power_x_off+shift, end_power_y_off, arrow=tk.LAST)

        interaction_muted_to_power_off_x = mid_temp_x2 
        interaction_muted_to_power_off_y = mid_temp_y2

        # interat with muted device <-> power on muted
        interact_muted_power_on_muted = canvas.create_line(start_x+5*shift, start_y, power_on_muted_x+5*shift, power_on_muted_y+height, arrow=tk.LAST)
        interaction_muted_power_on_muted_x1 = start_x+5*shift
        interaction_muted_power_on_muted_y1 = start_y
        interaction_muted_power_on_muted_x2 = power_on_muted_x+5*shift
        interaction_muted_power_on_muted_y2 = power_on_muted_y+height

        power_on_muted_interact_muted = canvas.create_line(power_on_muted_x+5.5*shift, power_on_muted_y+height, start_x+5.5*shift, start_y, arrow=tk.LAST)
        power_on_muted_interaction_muted_x1 = power_on_muted_x+5.5*shift 
        power_on_muted_interaction_muted_y1 = power_on_muted_y+height
        power_on_muted_interaction_muted_x2 = start_x + 5.5*shift
        power_on_muted_interaction_muted_y2 = start_y
        
    elif num==6: # interact with unmuted device
        x_p = x_pre - width/2
        y_p = y_pre + dis_box*2
        forward_event = canvas.create_line(start_x, start_y, end_x, end_y, arrow=tk.LAST)
        # backward arrow
        start_x = x_p + width/2
        start_y = y_p
        end_x = x_pre + width/2
        end_y = y_pre + height
        backward_event = canvas.create_line(end_x+shift, end_y, start_x+shift, start_y, arrow=tk.LAST)


        interaction_unmuted_to_interaction_muted_x1 = end_x+shift
        interaction_unmuted_to_interaction_muted_y1 = end_y
        interaction_unmuted_to_interaction_muted_x2 = start_x+shift
        interaction_unmuted_to_interaction_muted_y2 = start_y


        
        # interact with unmuted device -> power off
        interact_unmuted_off = canvas.create_line(start_x+width/2, start_y + height/2, start_x + 380, start_y-110, end_power_x_off+2*shift, end_power_y_off, arrow=tk.LAST)
        interaction_unmuted_to_power_off_x = start_x+380
        interaction_unmuted_to_power_off_y = start_y-110

        # interact with unmuted device <-> power on unmuted
        interact_unmuted_power_on_muted = canvas.create_line(start_x-width/2-3, start_y+height/2, start_x-200, start_y-200, power_on_unmuted_x, power_on_unmuted_y+height, arrow=tk.LAST )
        power_on_muted_interact_unmuted = canvas.create_line(power_on_unmuted_x + shift, power_on_unmuted_y+height, start_x-200+shift, start_y-200, start_x-width/2-3, start_y+height/2+shift, arrow=tk.LAST)


    elif num==7: # change volume on unmuted device
        x_p = x_pre - width
        y_p = y_pre + dis_box*2
        forward_event = canvas.create_line(start_x, start_y, end_x, end_y, arrow=tk.LAST)
        # backward arrow
        start_x = x_p + width/2
        start_y = y_p
        end_x = x_pre + width/2
        end_y = y_pre + height
        backward_event = canvas.create_line(end_x+shift, end_y, start_x+shift, start_y, arrow=tk.LAST)
        forward_event = canvas.create_line(start_x+2*shift, start_y, end_x+2*shift, end_y, arrow=tk.LAST)
        change_volume_unmuted_to_interaction_unmuted_x1 = end_x + shift
        change_volume_unmuted_to_interaction_unmuted_y1 = end_y
        change_volume_unmuted_to_interaction_unmuted_x2 = start_x + shift
        change_volume_unmuted_to_interaction_unmuted_y2 =  start_y

        # change volume on unmuted device -> power off
        change_volume_on_unmuted_off = canvas.create_line(start_x-width/2, start_y +height/2, start_x-250, start_y-400, end_power_x_off-width/2, end_power_y_off-height/2, arrow=tk.LAST)
        volume_change_unmuted_to_power_off_x = start_x-250
        volume_change_unmuted_to_power_off_y = start_y-400

        # change volume on unmuted device <-> power on unmuted
        change_volume_on_unmuted_power_on_unmuted = canvas.create_line(start_x, start_y, start_x, start_y-400, power_on_unmuted_x-width/2, power_on_unmuted_y+height/2, arrow=tk.LAST)
        change_volume_on_unmuted_power_on_unmuted_x = start_x
        change_volume_on_unmuted_power_on_unmuted_y = start_y-400

        power_on_unmuted_change_volume_on_unmuted = canvas.create_line(power_on_unmuted_x-width/2, power_on_unmuted_y+height/2, start_x-8*shift, start_y-450, start_x-8*shift, start_y, arrow=tk.LAST)
        power_on_unmuted_change_volume_on_unmuted_x = start_x-8*shift
        power_on_unmuted_change_volume_on_unmuted_y = start_y-450

        # change volume on unmuted device <-> change volume on muted
        change_volume_on_unmuted_change_volume_muted = canvas.create_line(start_x-shift, start_y, start_x-shift, start_y-300, change_volume_on_muted_x-width/2, change_volume_on_muted_y+height/2, arrow=tk.LAST)
        change_volume_on_unmuted_change_volume_muted_x = start_x-shift
        change_volume_on_unmuted_change_volume_muted_y = start_y-300

        change_volume_muted_change_volume_on_unmuted = canvas.create_line(change_volume_on_muted_x-width/2, change_volume_on_muted_y+height/2, start_x-5*shift, start_y-350, start_x-5*shift, start_y, arrow=tk.LAST)
        change_volume_muted_change_volume_on_unmuted_x = start_x-5*shift
        change_volume_muted_change_volume_on_unmuted_y = start_y-350


    prev_box = canvas.create_rectangle(x_p, y_p, x_p+width, y_p+height, fill='white')
    #states = ['power_on_unmuted', 'power_on_muted', 'change_volume_muted', 'interact_muted', 'interact_unmuted', 'change_volume_unmuted']

    if num==2: 
        power_on_unmuted_box = prev_box
    elif num==3:
        power_on_muted_box = prev_box
    elif num==4:
        change_volume_muted_box = prev_box
    elif num==5:
        interact_muted_box = prev_box
    elif num==6:
        interact_unmuted_box = prev_box
    elif num==7:
        change_volume_unmuted_box = prev_box
    x_pre = x_p
    y_pre = y_p

    #forward_event = canvas.create_line(start_x, start_y, end_x, end_y, arrow=tk.LAST)
    #backward_event = canvas.create_line(end_x+shift, end_y, start_x+shift, start_y, arrow=tk.LAST)

    canvas.update()

time.sleep(5)

#start_time = time.time()
# debugging
status = 0
prev_box = None
x = []
states = ['power_on_unmuted', 'power_on_muted', 'change_volume_muted', 'interact_muted', 'interact_unmuted', 'change_volume_unmuted']

for state in states:
    for file_num in range(1,16):
        """
        with open('./gh_data/network_features_' + state + str(file_num), 'rb') as network_f:
               network_d = pickle.load(network_f)
        """
        with open('./gh_data/power_features_'+state+str(file_num), 'rb') as power_f:
               power_d = pickle.load(power_f)
        with open('./gh_data/emanation_features_'+state+str(file_num), 'rb') as emanation_f:
               emanation_d = pickle.load(emanation_f)

        #x.append(np.array(network_d + power_d+emanation_d))
        x.append(np.array(power_d+emanation_d))
        
x = np.vstack(x)
scaler_x = StandardScaler()
scaler_x.fit(x)
x_scaled = scaler_x.transform(x)
tsne = TSNE(n_components=2, perplexity=5)
tsne_arr = tsne.fit_transform(x_scaled)

kmeans = KMeans(n_clusters=6)
kmeans.fit(tsne_arr)
pred_labels_x = kmeans.predict(tsne_arr)

labels = [0, 0, 0, 0, 0, 0]
y = [0]*15 + [1]*15 + [2]*15 + [3]*15 + [4]*15 + [5]*15
for i, c in enumerate(np.unique(pred_labels_x)):
       match_nums = [np.sum((pred_labels_x==c)*(y==t)) for t in np.unique(y)]
       print(match_nums)
       labels[i] = np.unique(y)[np.argmax(match_nums)]

print(labels)
status = 0
while(True):
    if True:
        #status +=1
        #time.sleep(5)
        print('hh')
    else:
       # ML model for state prediction
       # read pre-collected data
       while(True):
           # network traffic data capture and processing
           q = multiprocessing.Queue()

           #p1 = multiprocessing.Process(target=network_data.network_data, args=(q,))
           p2 = multiprocessing.Process(target=power_data.power_data, args=(q,))
           p3 = multiprocessing.Process(target=emanation_data.emanation_data, args=(q,))

           #p1.start()
           p2.start()
           p3.start()

           #p1.join()
           p2.join()
           p3.join()

           #network = q.get()
           power = q.get()
           emanation = q.get()

           # nine features for the data sets
           if len(power)>0:
               #if len(network) == 0:
               #    network = [0,0]

               features = [np.mean, np.var, lambda x: np.sqrt(np.mean(np.power(x, 2))), np.std, stats.median_abs_deviation, stats.skew, lambda x: stats.kurtosis(x, fisher=False), stats.iqr, lambda x: np.mean((x-np.mean(x))**2)]
               fea_power = [feature(power) for feature in features]
               #fea_network = [feature(network) for feature in features]
               fea_emanation = [feature(emanation) for feature in features]
               #fea = np.array(fea_power+fea_network+fea_emanation)
               fea = np.array(fea_power+fea_emanation)


               fea = fea.reshape(1, fea.shape[0])
               sample_transform = scaler_x.transform(fea)
               stacked = np.vstack([x_scaled, sample_transform])
               tsne_arr = tsne.fit_transform(stacked)
               pred_label =  kmeans.fit_predict(tsne_arr[:-1])
               labels = [0, 0, 0, 0, 0,0]
               y = [0]*15 + [1]*15 + [2]*15 + [3]*15 + [4]*15 + [5]*15
               for i, c in enumerate(np.unique(pred_label)):
                   match_nums = [np.sum((pred_label==c)*(y==t)) for t in np.unique(y)]
                   print(match_nums)
                   labels[i] = np.unique(y)[np.argmax(match_nums)]
               last_row = tsne_arr[-1,:]
               last_row = last_row.reshape(1, last_row.shape[0])
               samp_label = kmeans.predict(last_row)
               if samp_label[0] not in labels:
                   status = 1
               else:
                   status = labels.index(samp_label[0])+1
               #status = labels.index(samp_label[0]) + 1
               break
    status +=1
    time.sleep(5)
    if prev_box is not None: 
            canvas.itemconfig(prev_box, fill='white')
    #states = ['power_on_unmuted', 'power_on_muted', 'change_volume_muted', 'interact_muted', 'interact_unmuted', 'change_volume_unmuted']
    if status==1: # power_on_unmuted
        canvas.itemconfig(power_on_unmuted_box, fill='light blue')
        x1_cur, y1_cur, x2_cur, y2_cur = canvas.coords(power_on_unmuted_box)
        canvas.create_text((x1_cur+x2_cur)/2, (y1_cur+y2_cur)/2, text="power_on_unmuted", font=("Times New Roman ", 15)) 

       
        text_xp = (power_off_to_power_on_unmuted_x1+power_off_to_power_on_unmuted_x2)/2 - 40
        text_yp = (power_off_to_power_on_unmuted_y1+power_off_to_power_on_unmuted_y2)/2
        canvas.create_text(text_xp+50, text_yp+10, text="plug in", font=("Times New Roman", 15)) # starting the device
      
        text_xp = (power_on_unmuted_to_power_off_x1+power_on_unmuted_to_power_off_x2)/2 + 10
        text_yp = (power_on_unmuted_to_power_off_y1+power_on_unmuted_to_power_off_y2)/2 + 10
        canvas.create_text(text_xp-50, text_yp-20, text="unplug", font=("Times New Roman", 15)) # starting the device

        canvas.create_text(x1_cur+60, y1_cur+400, text="interaction", font=("Times New Roman", 15))
 
        prev_box = power_on_unmuted_box
        canvas.update()
    elif status==2: #  power on muted
        canvas.itemconfig(power_on_muted_box, fill='light blue')
        x1_cur, y1_cur, x2_cur, y2_cur = canvas.coords(power_on_muted_box)
        canvas.create_text((x1_cur+x2_cur)/2, (y1_cur+y2_cur)/2, text="power_on_muted", font=("times new roman ", 15)) 
        x1_pre, y1_pre, x2_pre, y2_pre = canvas.coords(power_on_muted_box)

        #text_xp = (power_on_unmuted_to_power_on_muted_x1 + power_on_unmuted_to_power_on_muted_x2)/2
        #text_yp = (power_on_unmuted_to_power_on_muted_y1 + power_on_unmuted_to_power_on_muted_y2)/2
        #canvas.create_text(text_xp, text_yp, text="press mute button", font=("times new roman", 15)) # pressing the button

        text_xp = (power_on_muted_to_power_on_unmuted_x1 + power_on_muted_to_power_on_unmuted_x2)/2
        text_yp = (power_on_muted_to_power_on_unmuted_y1 + power_on_muted_to_power_on_unmuted_y2)/2
        canvas.create_text(text_xp, text_yp, text="press button", font=("times new roman", 15)) # pressing the button
        canvas.create_text(power_on_muted_to_power_off_x+20, power_on_muted_to_power_off_y-20, text="unplug", font=("times new roman", 15)) # pressing the button


        prev_box = power_on_muted_box
        canvas.update()
    elif status==3: # change_volume_muted
        canvas.itemconfig(change_volume_muted_box, fill='light blue')
        x1_cur, y1_cur, x2_cur, y2_cur = canvas.coords(change_volume_muted_box)
        canvas.create_text((x1_cur+x2_cur)/2, (y1_cur+y2_cur)/2, text="change_volume_muted", font=("times new roman ", 15)) 
        x1_pre, y1_pre, x2_pre, y2_pre = canvas.coords(change_volume_muted_box)
        text_xp = (change_volume_muted_to_power_on_muted_x1 + change_volume_muted_to_power_on_muted_x2)/2
        text_yp = (change_volume_muted_to_power_on_muted_y1 + change_volume_muted_to_power_on_muted_y2)/2
        canvas.create_text(text_xp, text_yp, text="volume change", font=("times new roman", 15)) # quering the device
        canvas.create_text(change_volume_muted_to_power_off_x, change_volume_muted_to_power_off_y+30, text="unplug", font=("times new roman", 15)) # quering the device
        

        prev_box = change_volume_muted_box
        canvas.update()
    elif status==4: #  interact muted 
        canvas.itemconfig(interact_muted_box, fill='light blue')
        x1_cur, y1_cur, x2_cur, y2_cur = canvas.coords(interact_muted_box)
        canvas.create_text((x1_cur+x2_cur)/2, (y1_cur+y2_cur)/2, text="interaction_muted", font=("times new roman ", 15)) 
        x1_pre, y1_pre, x2_pre, y2_pre = canvas.coords(interact_muted_box)

        text_xp =  (interaction_muted_to_change_volume_muted_x1 + interaction_muted_to_change_volume_muted_x2)/2
        text_yp =  (interaction_muted_to_change_volume_muted_y1 + interaction_muted_to_change_volume_muted_y2)/2
        canvas.create_text(text_xp-65, text_yp, text="interaction", font=("times new roman", 15)) # finishing processing
        
        canvas.create_text(text_xp+55, text_yp-5, text="volume change", font=("times new roman", 15)) # finishing processing
        canvas.create_text(interaction_muted_to_power_off_x-50, interaction_muted_to_power_off_y, text="unplug", font=("times new roman", 15)) # finishing processing



        prev_box = interact_muted_box
        canvas.update()
    elif status==5: # interact unmuted
        canvas.itemconfig(interact_unmuted_box, fill='light blue')
        x1_cur, y1_cur, x2_cur, y2_cur = canvas.coords(interact_unmuted_box)
        canvas.create_text((x1_cur+x2_cur)/2, (y1_cur+y2_cur)/2, text="interaction_unmuted", font=("times new roman ", 15)) 
        x1_pre, y1_pre, x2_pre, y2_pre = canvas.coords(interact_unmuted_box)
        text_xp =  (interaction_unmuted_to_interaction_muted_x1 + interaction_unmuted_to_interaction_muted_x2)/2
        text_yp =  (interaction_unmuted_to_interaction_muted_y1 + interaction_unmuted_to_interaction_muted_y2)/2
        canvas.create_text(text_xp, text_yp, text="press button", font=("times new roman", 15)) # finishing processing
        canvas.create_text(text_xp-220, text_yp-330, text="finish response", font=("times new roman", 15)) # finishing processing
        canvas.create_text(interaction_unmuted_to_power_off_x, interaction_unmuted_to_power_off_y+25, text="unplug", font=("times new roman", 15) ) 
        canvas.create_text((interaction_muted_power_on_muted_x1+interaction_muted_power_on_muted_x2)/2 + 65, (interaction_muted_power_on_muted_y1+interaction_muted_power_on_muted_y2)/2+15, text="finish response", font=("times new roman", 15) ) 
        canvas.create_text((power_on_muted_interaction_muted_x1+power_on_muted_interaction_muted_x2)/2+ 25, (power_on_muted_interaction_muted_y1+power_on_muted_interaction_muted_y2)/2 - 10, text="touch", font=("times new roman", 15))
        prev_box = interact_unmuted_box
        canvas.update()
    elif status==6: # change volume unmuted
        canvas.itemconfig(change_volume_unmuted_box, fill='light blue')
        x1_cur, y1_cur, x2_cur, y2_cur = canvas.coords(change_volume_unmuted_box)
        canvas.create_text((x1_cur+x2_cur)/2, (y1_cur+y2_cur)/2, text="change_volume_unmuted", font=("times new roman ", 15)) 
        x1_pre, y1_pre, x2_pre, y2_pre = canvas.coords(change_volume_unmuted_box)
        text_xp = (change_volume_unmuted_to_interaction_unmuted_x1 + change_volume_unmuted_to_interaction_unmuted_x2)/2
        text_yp =  (change_volume_unmuted_to_interaction_unmuted_y1 + change_volume_unmuted_to_interaction_unmuted_y2)/2
        canvas.create_text(text_xp+45, text_yp+10, text="interaction", font=("times new roman", 15)) # finishing processing
        canvas.create_text(text_xp-50, text_yp-15, text="volume change", font=("times new roman", 15)) # finishing processing
        canvas.create_text(volume_change_unmuted_to_power_off_x+40, volume_change_unmuted_to_power_off_y, text="unplug", font=("times new roman", 15))
        canvas.create_text(change_volume_on_unmuted_power_on_unmuted_x, change_volume_on_unmuted_power_on_unmuted_y, text="volume change", font=("times new roman", 15))
        canvas.create_text(power_on_unmuted_change_volume_on_unmuted_x, power_on_unmuted_change_volume_on_unmuted_y, text="volume change", font=("times new roman", 15))
        canvas.create_text(change_volume_on_unmuted_change_volume_muted_x, change_volume_on_unmuted_change_volume_muted_y, text="press button", font=("times new roman", 15))
        canvas.create_text(change_volume_muted_change_volume_on_unmuted_x, change_volume_muted_change_volume_on_unmuted_y, text="press button", font=("times new roman", 15))


        prev_box = change_volume_unmuted_box
        canvas.update()
    else:
        #canvas.itemconfig(listen_box, fill='light blue')
        #prev_box = listen_box
        status = 0
        print("finishing debugging!")
        canvas.update()
window.mainloop()
