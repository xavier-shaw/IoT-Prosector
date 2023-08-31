from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
from typing import List
from dotenv import dotenv_values
from pymongo import MongoClient
from urllib.parse import quote_plus
import certifi
import numpy as np
import time
import pickle
import multiprocessing
import network_data
import power_data
import emanation_data
from scipy import stats
from sklearn.preprocessing import StandardScaler
import os
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import json
import threading
import serial
from paramiko import SSHClient

config = dotenv_values(".env")
username = quote_plus(config["NAME"])
password = quote_plus(config["PASSWORD"])
cluster = config["CLUSTER"]
uri = 'mongodb+srv://' + username + ':' + password + \
    '@' + cluster + '/?retryWrites=true&w=majority'
pineapple_token = "eyJVc2VyIjoicm9vdCIsIkV4cGlyeSI6IjIwMjgtMDgtMjJUMDI6Mzc6NTUuMjk4NzgzMzAzWiIsIlNlcnZlcklkIjoiYTYyMTM3MzE3NTUyNDRlZSJ9.sbCLEXl3vWayXfd4zM2zgTthQnzEztZvWxNi_nejdvg="


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================================== GLOBAL VARIABLES ==================================================
ser = None
listening = False
data_thread = None

max_currents = []
avg_currents = []
min_currents = []
times = []
start_time = 0

centroids = {}
sample_num = 150
group_cnt = 10

def read_from_arduino():
    print("Start reading from Arduino...")
    global ser
    ser = serial.Serial('/dev/tty.usbmodem21101', 9600, timeout=1)
    global avg_currents, max_currents, min_currents, times, listening
    while True:
        line = ser.readline()
        if line and listening:
            info = line.decode().rstrip()
            infos = info.split(",")
            max_current = float(infos[0])
            avg_current = float(infos[1])
            min_current = float(infos[2])
            avg_currents.append(avg_current)
            max_currents.append(max_current)
            min_currents.append(min_current)
            times.append(time.time() - start_time)

data_thread = threading.Thread(target=read_from_arduino, daemon=True)
data_thread.start()


@app.on_event("startup")
async def startup_db_client():
    app.client = MongoClient(uri, tlsCAFile=certifi.where())
    app.database = app.client[config["DB_NAME"]]
    print("Connected to the MongoDB database!")
    print(app.client)
    print(app.database)

    # print("ready to ssh")
    # ssh = SSHClient()
    # ssh.load_system_host_keys()
    # ssh.connect(hostname="172.16.42.1", username="root", password="hak5pineapple")
    # ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command("tcpdump -i any -w filename.pcap tcp and host 172.16.42.205")
    # print(ssh_stdout)


@app.on_event("shutdown")
def shutdown_db_client():
    global data_thread, ser
    data_thread.join()
    ser.close()
    app.client.close()

# ========================================= Routes =========================================================


@app.get("/")
async def root():
    return {"message": "Here is the hardware end of IoT-Auditor!"}


@app.get("/get_data")
async def get_data():
    # devices = ["listening keyword", "listening monitor speech",
    #            "muted tap", "muted monitor speech"]
    # devices = ['Keyword Listening 08.15', 'Keyword Listening Casting 08.15', 'Command Listening 08.15', 'Command Listening Casting 08.15', 'Unmuted Playing Muted 08.15', 'Playing Vol Min 08.15', 'Playing Vol Mid 08.15', 'Playing Vol Max 08.15',
    #            'Muted Base 08.15', 'Muted Base Casting 08.15', 'Muted Playing Muted 08.15', 'Muted Playing Vol Min 08.15', 'Muted Playing Vol Mid 08.15', 'Muted Playing Vol Max 08.15',
    #            'Keyword Listening 08.16', 'Keyword Listening Max 08.16', 'Muted Base 08.16', 'Muted Base Max 08.16']
    devices = ['Keyword Listening 08.15', 'Keyword Listening Casting 08.15', 'Playing Vol Min 08.15', 'Playing Vol Max 08.15',
               'Muted Base 08.15', 'Muted Base Casting 08.15', 'Muted Playing Vol Min 08.15', 'Muted Playing Vol Max 08.15']

    centers = []
    radiuses = []
    features = []
    datas = []
    point_state_dict = {}
    point_labels = []
    idx = 0
    for device in devices:
        data = app.database["iotdatas"].find({"device": device})
        center, radius, feas = compute_data_features(data)
        centers.append(center)
        radiuses.append(radius)
        datas.append(feas)
        for fea in feas:
            features.append(fea[0])
            point_state_dict[idx] = device
            point_labels.append(devices.index(device))
            idx += 1

    for i in range(len(devices)):
        now_device = devices[i]
        now_center = centers[i]
        now_radius = radiuses[i]
        now_datas = datas[i]
        print(now_device + " (" + str(now_radius) + "):")
        for j in range(i + 1, len(devices)):
            cur_device = devices[j]
            cur_datas = datas[j]
            center_center_distance = calculate_distance(centers[j], now_center)
            # print(now_device + " -> " + cur_device +
            #       ": " + str(pairwise_distance))
            point_center_distance = calculate_pairwise_distance(
                now_datas, centers[j])
            print(format(center_center_distance, ".2f"),
                  ' , ', format(point_center_distance, ".2f"))
        print("================================================")

    draw(devices, features, point_state_dict, point_labels)

    return {"message": "finish data processing"}


@app.get("/check")
async def check():
    device = "test data 4"
    datas = app.database["iotdatas"].find({"device": device})
    powers = []
    power_timestamps = []
    emanations = []
    emanation_timestamps = []
    features = []
    for data in datas:
        powers.append(data["power"])
        power_timestamps.append(data["power_timestamp"])
        emanation = data["emanation"]
        emanation_timestamps.append(data["emanation_timestamp"][0])

    powers = np.concatenate(powers)
    power_timestamps = np.concatenate(power_timestamps)

    # Create a new figure and axis

    # Plot the first line chart on the first y-axis
    plt.scatter(power_timestamps, powers)

    # # Create a second y-axis that shares the same x-axis
    # ax2 = ax1.twinx()

    # # Plot the second line chart on the second y-axis
    # ax2.plot(times, emanations, 'b-')
    # ax2.set_ylabel('Emanation', color='b')
    # ax2.tick_params('y', colors='b')

    # Set the title
    plt.title("Power Changes with Time")
    plt.savefig("power_time.png")

    return {"message": "verify state changes"}


@app.get("/getBoards")
async def get_boards():
    boards = app.database["boards"].find()
    titles = [board["title"] for board in boards]
    return {"message": str(titles)}


@app.get("/start/")
async def start_sensing(device: str):
    global isSensing
    isSensing = True
    # if loop_thread is None or not loop_thread.is_alive():
    #     loop_thread = threading.Thread(target=sensing(device))
    #     loop_thread.start()
    #     return {"message": "Sensing IoT device started: " + device}
    # else:
    #     return {"message": "Sensing IoT device is already running"}
    sensing(device)
    return {"message": "start sensing " + device}


@app.get("/prep/")
async def prep_for_predict(device: str):
    cal_clusters_center(device)
    return {"message": "Here"}


@app.get("/predict/")
async def annotate_sensing(device: str):
    global isSensing
    isSensing = True
    if loop_thread is None or not loop_thread.is_alive():
        loop_thread = threading.Thread(target=predict_sensing(device))
        loop_thread.start()
        return {"message": "Predicting IoT device state: " + device}
    else:
        return {"message": "Predicting IoT device state is already running"}


@app.get("/startlocal/")
async def start_local_sensing(device: str):
    local_sensing(device)
    return {"message": "Sensing IoT device from local file: " + device}


@app.get("/remove/")
async def remove(device: str):
    app.database["iotdatas"].delete_many({"device": device})
    app.database["actionvideos"].delete_many({"device": device})
    app.database["boards"].delete_many({"title": device})
    return {"message": "Delete all data of " + device}


@app.get("/removeAllVideos")
async def remove_all_data():
    app.database["actionvideos"].delete_many({})
    return {"message": "Delete all video."}


@app.get("/removeAllData")
async def remove_all_data():
    # app.database["iotstates"].delete_many({})
    app.database["iotdatas"].delete_many({})
    return {"message": "Delete all data."}


@app.get("/removeAllBoards")
async def remove_all_boards():
    app.database["boards"].delete_many({})
    return {"message": "delete all boards."}


@app.get("/startSensing")
async def start_sensing():
    global listening, start_time
    listening = True
    start_time = time.time()
    return {"message": "Start Recording!"}


@app.get("/stopSensing")
async def stop_sensing(idx: str, device: str):
    global listening, max_currents, avg_currents, min_currents, times
    listening = False
    store_power_data(device, idx, max_currents,
                     avg_currents, min_currents, times)
    max_currents = []
    avg_currents = []
    min_currents = []
    times = []

    return {"message": "Stop Recording!"}


class DataModel(BaseModel):
    device: str
    nodes: List[dict]

@app.post("/classification")
async def classfication(data: DataModel = Body(...)):
    states, X, Y = build_dataset(data.device, data.nodes)
    clf, cms, acc = classify(states, X, Y)
    resp = {
        "accuracy": round(acc, 3),
        "confusionMatrix": cms,
        "states": states
    }

    return JSONResponse(content=jsonable_encoder(resp))
# ========================================= Functions =========================================================


def power_check():
    avg_currents, max_currents, min_currents, times = [], [], [], []
    start_time = time.time()

    for i in range(sample_num):
        line = ser.readline()
        print(line)
        if line:
            info = line.decode().rstrip()
            infos = info.split(",")
            max_current = float(infos[0])
            avg_current = float(infos[1])
            min_current = float(infos[2])
            avg_currents.append(avg_current)
            max_currents.append(max_current)
            min_currents.append(min_current)
            times.append(time.time() - start_time)

    print("duration: ", str(times[len(times) - 1] - times[0]))
    return avg_currents, max_currents, min_currents, times


def store_power_data(device, idx, max_currents, avg_currents, min_currents, times):
    data = {
        "device": device,
        "idx": idx,
        "max_currents": max_currents,
        "avg_currents": avg_currents,
        "min_currents": min_currents,
        "times": times
    }
    create_data(jsonable_encoder(data))


def build_dataset(device, nodes):
    collected_node = []
    labels = []
    dataset_X = []
    dataset_Y = []

    group_nodes = [n for n in nodes if n["type"] == "modeNode"]
    state_nodes = [n for n in nodes if n["type"] == "stateNode"]

    for group_node in group_nodes:
        children = group_node["data"]["children"]
        if len(children) > 0:
            labels.append(group_node["data"]["label"])
            for child in children:
                collected_node.append(child)
                features = get_features(device, child)
                for feature in features:
                    dataset_X.append(feature)
                    dataset_Y.append(len(labels) - 1)

    for state_node in state_nodes:
        if state_node["id"] not in collected_node:
            collected_node.append(state_node["id"])
            labels.append(state_node["data"]["label"])
            features = get_features(device, state_node["id"])
            for feature in features:
                dataset_X.append(feature)
                dataset_Y.append(len(labels) - 1)

    dataset_X = np.array(dataset_X)
    dataset_Y = np.array(dataset_Y)
    return labels, dataset_X, dataset_Y


def get_features(device, idx):
    avg_currents, max_currents, min_currents, times = get_data(device, idx)
    group_size = 5
    features = []

    for i in range(len(avg_currents)):
        start_idx = i * group_size
        if start_idx >= len(avg_currents): break
        end_idx = (i + 1) * group_size if (i + 1) * group_size < len(avg_currents) else len(avg_currents)
        avgs = avg_currents[start_idx: end_idx]
        maxs = max_currents[start_idx: end_idx]
        mins = min_currents[start_idx: end_idx]
        avg_features = [fea(avgs) for fea in [
            np.mean, np.std, lambda x: np.max(x) - np.min(x)]]
        max_features = [fea(maxs) for fea in [np.mean, np.std]]
        min_features = [fea(mins) for fea in [np.mean, np.std]]
        feas = np.concatenate(
            (avg_features, max_features, min_features))
        features.append(feas)

    return features


def get_data(device, state):
    data = app.database["iotdatas"].find_one({"device": device, "idx": state})
    print(data)
    avg_currents = data["avg_currents"]
    max_currents = data["max_currents"]
    min_currents = data["min_currents"]
    times = data["times"]

    return avg_currents, max_currents, min_currents, times


def classify(states, X, Y):
    # clf = LogisticRegression()
    clf = DecisionTreeClassifier()
    # clf = RandomForestClassifier(n_estimators=30, random_state=42)

    kf = KFold(n_splits=10, shuffle=True)

    cms = np.zeros((len(states), len(states)))
    acc_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=[
                              i for i in range(len(states))])
        cms += cm

        acc = accuracy_score(y_test, y_pred)
        acc_scores.append(acc)

    avg_accuracy = np.mean(acc_scores)
    avg_cm = []
    for row in cms:
        row_sum = np.sum(row)
        avg_cm.append([format(x / row_sum, ".2f") for x in row])

    return clf, avg_cm, avg_accuracy


# ===========================================================================================================


def calculate_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))  # Euclidean distance


def calculate_centroid(cluster):
    return np.mean(cluster, axis=0)


def get_closest_centroid(point, centroids):
    distances = [calculate_distance(point, centroid) for centroid in centroids]
    return np.argmin(distances), np.min(distances)


def predict_closest_centroid(point):
    global centroids
    min_distance = 1e9
    current_state = ""
    for state, center in centroids.items():
        distance = calculate_distance(point, center)
        if distance < min_distance:
            min_distance = distance
            current_state = state

    return current_state


def create_state(state):
    app.database["iotstates"].insert_one(state)
    print("==================== NEW STATE =====================")
    print(state)
    print("==================== NEW STATE =====================")


def create_data(data):
    app.database["iotdatas"].insert_one(data)
    print("store data into database")


def sensing(device):
    print("\n")
    start_time = time.time()

    data_points_info = {}
    data_point_idx = 0
    state_clusters = []  # identified clusters
    centroids = []  # center point of clusters
    distance_threshold = 1.2  # threshold for outlier
    previous_data_cluster_idx = -1  # the state of previous data
    # threshold for new cluster (times of continous outlier)
    count_threshold = 2
    outlier_buffer = []  # a buffer array for potential new cluster
    outlier_buffer_idx = []  # an array records the potential outliers' idx
    new_state = True  # indicator for creating a new state
    count = 0

# ======================= Read data from data stream ========================================
    isSensing = True
    first = True  # the first data always wrong
    while (isSensing and count < 40):
        # sensing_variable = app.database["sharedvariables"].find_one(
        #     {"name": "sensing"})
        # if sensing_variable["value"] == "false":
        #     print("stop sensing at exploration stage!!!!!")
        #     isSensing = False
        #     break

        q = multiprocessing.Queue()
        # p1 = multiprocessing.Process(target=network_data.network_data, args=(q,))
        p2 = multiprocessing.Process(
            target=power_data.power_data, args=(q, start_time))
        p3 = multiprocessing.Process(
            target=emanation_data.emanation_data, args=(q, start_time))

        # p1.start()
        p2.start()
        p3.start()
        # p1.join()
        p2.join()
        p3.join()
        # n = q.get()
        e = q.get()
        t2 = q.get()
        p = q.get()
        t1 = q.get()
        # networks.append(n)
        if len(p) > 0:
            # skip the first data
            if first:
                first = False
                # Create Default POWER_OFF state as the first state
                new_state_info = {
                    "time": time.time() - start_time,
                    "device": device,
                    "idx": "-1",
                    "prev_idx": "-99"
                }
                create_state(new_state_info)
                continue

            count += 1
            print("power: ", p)
            print("power timestamps: ", t1)
            print("emanation: ", e)
            print("emanation timestamps", t2)
            print("time: ", time.time() - start_time)
            print("data point count: " + str(count))

            features = [np.mean, np.var, lambda x: np.sqrt(np.mean(np.power(x, 2))), np.std, stats.median_abs_deviation, stats.skew, lambda x: stats.kurtosis(
                x, fisher=False), stats.iqr, lambda x: np.mean((x-np.mean(x))**2)]
            # fea_network = [feature(network) for feature in features]
            fea_power = [feature(p) for feature in features]
            fea_emanation = [feature(e) for feature in features]
            fea = np.array(fea_power + fea_emanation)
            fea = fea.reshape(1, fea.shape[0])

            # Clustering Workflow:
            # 1. calculate distance between data point and clusters's center points
            # 2. if distance larger than threshold, it is an "outlier":
            #       (a) if cumulated outlier count larger than count threshold => create new cluster for cumulated outliers
            #       (b) if cumulated outlier count less than count threshold => record the outlier in a buffer
            # 3. if distance smaller than threshold:
            #       (a) if the nearest cluster is the previous cluster => add data point to the nearest cluster
            #       (b) if the nearest cluster is another cluster => a new threshold to judge if add to it or not? => just add to the nearest cluster
            #    => recalculate the center point of the modified cluster
            #    => clear the outlier buffer to confirm they're outliers

            if len(centroids) == 0:  # if this is the first point
                state_clusters.append([fea])
                centroids.append(fea)
                cluster_idx = len(state_clusters) - 1
                new_state = True
            else:
                # calculate distance
                closest_centroid_index, closest_distance = get_closest_centroid(
                    fea, centroids)
                print("new data point's closest distance: ", closest_distance)
                print("now state: " + str(previous_data_cluster_idx))
                # less than threshold
                if closest_distance <= distance_threshold:
                    print("smaller than threshold")
                    # the nearest cluster is current state
                    if closest_centroid_index == previous_data_cluster_idx:
                        belonged_cluster_idx = closest_centroid_index
                    # the nearest cluster is not current state
                    else:
                        belonged_cluster_idx = closest_centroid_index
                        new_state = True
                    state_clusters[belonged_cluster_idx].append(fea)
                    # recalculate the center point
                    centroids[belonged_cluster_idx] = calculate_centroid(
                        state_clusters[belonged_cluster_idx])
                    # empty the outlier buffer and its idx
                    outlier_buffer = []
                    outlier_buffer_idx = []
                    cluster_idx = belonged_cluster_idx
                    print("Next state is: " + str(belonged_cluster_idx))
                # larger than threshold
                else:
                    print("larger than threshold")
                    print("outlier buffer count: " +
                          str(len(outlier_buffer) + 1))
                    # add to outlier buffer
                    outlier_buffer.append(fea)
                    # number of outliers more than the threshold => create new cluster
                    if len(outlier_buffer) >= count_threshold:
                        # add the outlier buffer as a new cluster
                        state_clusters.append(outlier_buffer)
                        # calculate the center point of the new cluster
                        centroids.append(calculate_centroid(outlier_buffer))
                        new_state = True
                        cluster_idx = len(state_clusters) - 1
                        # update the outliers' state as this new cluster
                        for outlier_idx in outlier_buffer_idx:
                            data_points_info[outlier_idx]["state"] = str(
                                cluster_idx)
                        # empty the outlier buffer and its idx
                        outlier_buffer = []
                        outlier_buffer_idx = []
                    # number of outliers less than the threshold
                    else:
                        # indicate this data point is an outlier
                        cluster_idx = previous_data_cluster_idx
                        # add the idx to the buffer so that its state can be updated later
                        outlier_buffer_idx.append(data_point_idx)

            data_point_info = {
                "idx": str(data_point_idx),
                "state": str(cluster_idx),
                "data": fea.tolist(),
                "time": time.time() - start_time,
                "device": device,
                "power": np.array(p).tolist(),
                "power_timestamp": t1,
                "emanation": np.array(e).tolist(),
                "emanation_timestamp": t2
            }
            data_points_info[data_point_idx] = data_point_info
            data_point_idx += 1
            create_data(jsonable_encoder(data_point_info))

            if new_state:
                new_state_info = {
                    "time": time.time() - start_time,
                    "device": device,
                    "idx": str(cluster_idx),
                    "prev_idx": str(previous_data_cluster_idx)
                }
                create_state(new_state_info)
                new_state = False
                previous_data_cluster_idx = cluster_idx  # record the current state

    # Store data in database
    # for data_point in data_points_info.values():
    #     create_data(jsonable_encoder(data_point))


def predict_sensing(device):
    while (True):
        sensing_variable = app.database["sharedvariables"].find_one(
            {"name": "sensing"})
        if sensing_variable["value"] == "false":
            print("stop sensing at annotation stage!!!!!")
            break
        powers = []
        emanations = []
        times = 0
        while times < 3:
            q = multiprocessing.Queue()
            # p1 = multiprocessing.Process(target=network_data.network_data, args=(q,))
            p2 = multiprocessing.Process(
                target=power_data.power_data, args=(q,))
            p3 = multiprocessing.Process(
                target=emanation_data.emanation_data, args=(q,))

            # p1.start()
            p2.start()
            p3.start()
            # p1.join()
            p2.join()
            p3.join()
            # n = q.get()
            p = q.get()
            e = q.get()
            # networks.append(n)
            if len(p) > 0:
                features = [np.mean, np.var, lambda x: np.sqrt(np.mean(np.power(x, 2))), np.std, stats.median_abs_deviation, stats.skew, lambda x: stats.kurtosis(
                    x, fisher=False), stats.iqr, lambda x: np.mean((x-np.mean(x))**2)]
                # fea_network = [feature(network) for feature in features]
                fea_power = [feature(p) for feature in features]
                fea_emanation = [feature(e) for feature in features]
                if powers == []:
                    powers = fea_power
                    emanations = fea_emanation
                else:
                    powers = np.vstack((powers, fea_power))
                    emanations = np.vstack((emanations, fea_emanation))
                times += 1
                print("data points: " + str(times) + "/3")

        fea_power_mean = np.mean(powers, axis=0)
        fea_emanation_mean = np.mean(emanations, axis=0)
        fea = np.hstack((fea_power_mean, fea_emanation_mean))

        current_state = predict_closest_centroid(fea)
        predict_state = {
            "device": device,
            "state": current_state
        }
        # app.database["predictstates"].delete_many({"device": device})
        app.database["predictstates"].insert_one(
            jsonable_encoder(predict_state))

    app.database["predictstates"].delete_many({"device": device})


# ======================= Read data from static files =======================================


def local_sensing(device):
    print("\n")
    start_time = time.time_ns()
    absolute_path = os.path.dirname(__file__)
    relative_path = "data"
    full_path = os.path.join(absolute_path, relative_path)

    states = ['power_on_unmuted', 'power_on_muted', 'change_volume_muted',
              'interact_muted', 'interact_unmuted', 'change_volume_unmuted']

    points_data = []
    data_points_info = []
    data_point_idx = 0
    state_clusters = []  # identified clusters
    centroids = []  # center point of clusters
    distance_threshold = 10000000
    new_state = True  # indicator just for testing
    previous_data_cluster_idx = -1  # the state of previous data

    for state in states:
        for file_num in range(1, 16):
            data_point_info = {}  # record the infomation of this data point
            power_file_name = 'power_features_' + state + str(file_num)
            power_data_path = os.path.join(full_path, power_file_name)
            # emalation_file_name = 'emalation_features_' + state + str(file_num)
            # emalation_data_path = os.path.join(full_path, emalation_file_name)

            with open(power_data_path, 'rb') as power_f:
                power_d = pickle.load(power_f)
            # with open(emalation_data_path, 'rb') as emanation_f:
            #     emanation_d = pickle.load(emanation_f)

            power_data = np.array(power_d)
            points_data.append(power_data)

            # Workflow:
            # 1. calculate distance between data point and clusters's center points
            # 2. if distance larger than threshold => create new cluster;
            # 3. if distance smaller than threshold => add data point to the nearest cluster => recalculate the center point of that cluster

            if len(centroids) == 0 or new_state:  # if this is the first point
                state_clusters.append([power_data])
                centroids.append(power_data)
                cluster_idx = len(state_clusters) - 1
            else:
                closest_centroid_index, closest_distance = get_closest_centroid(
                    power_data, centroids)
                if closest_distance < distance_threshold:
                    # if distance less than threshold, then the state won't change?
                    closest_centroid_index = previous_data_cluster_idx
                    state_clusters[closest_centroid_index].append(power_data)
                    centroids[closest_centroid_index] = calculate_centroid(
                        state_clusters[closest_centroid_index])
                    cluster_idx = closest_centroid_index
                else:
                    state_clusters.append([power_data])
                    centroids.append(power_data)
                    cluster_idx = len(state_clusters) - 1
                    new_state = True

            data_point_info = {
                "idx": str(data_point_idx),
                "state": str(cluster_idx),
                "data": power_data.tolist(),
                "time": time.time_ns() - start_time,
                "device": device
            }
            data_point_idx += 1
            data_points_info.append(data_point_info)

            if new_state:
                new_state_info = {
                    "time": time.time_ns() - start_time,
                    "device": device,
                    "idx": str(cluster_idx),
                    "prev_idx": str(previous_data_cluster_idx)
                }
                create_state(new_state_info)
                new_state = False
                previous_data_cluster_idx = cluster_idx  # record the current state

        new_state = True

    scaler_x = StandardScaler()
    transformed_points_data = scaler_x.fit_transform(points_data)
    # TSNE
    tsne = TSNE(n_components=2)
    tsne_transformed_data = tsne.fit_transform(transformed_points_data)

    for idx in range(len(data_points_info)):
        data_point = data_points_info[idx]
        data_point["tranformed_data"] = transformed_points_data[idx].tolist()
        data_point["tsne_data"] = tsne_transformed_data[idx].tolist()
        create_data(jsonable_encoder(data_point))


def compute_data_features(data):
    features = []
    mean_arr = []
    count = 1
    for d in data:
        mean_arr.append(d["data"])
        count -= 1
        if count == 0:
            mean_fea = np.mean(mean_arr, axis=0)
            mean_arr = []
            count = 1
            features.append(mean_fea)

    center_point = calculate_centroid(features)
    total_distance = 0
    for data_point in features:
        distance = calculate_distance(data_point, center_point)
        total_distance += distance

    radius = format(total_distance / len(features), ".2f")

    return center_point, radius, features


def draw(devices, features, point_state_dict, point_labels):
    scaler = StandardScaler()
    feature_scaled = scaler.fit_transform(features)

    # ======================= Elbow Method & Silhouette Score ============================
    wcss = []
    for i in range(1, len(devices)):  # Testing for up to 10 clusters
        kmeans = KMeans(n_clusters=i, init='k-means++',
                        max_iter=300, n_init=10, random_state=0)
        kmeans.fit(feature_scaled)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(devices)), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')

    # Silhouette Score
    sil = []
    for i in range(2, len(devices)):  # Silhouette score is defined only for more than 1 cluster
        kmeans = KMeans(n_clusters=i, init='k-means++',
                        max_iter=300, n_init=10, random_state=0)
        kmeans.fit(feature_scaled)
        silhouette_avg = silhouette_score(feature_scaled, kmeans.labels_)
        sil.append(silhouette_avg)

    plt.subplot(1, 2, 2)
    plt.plot(range(2, len(devices)), sil)
    plt.title('Silhouette Score Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')

    plt.tight_layout()
    plt.savefig('silhouette_score_1.png', bbox_inches='tight')

    # # TSNE + KMEANS CLUSTER VISUALIZATION
    # tsne = TSNE(n_components=2, perplexity=40, init="pca")
    # tsne_features = tsne.fit_transform(feature_scaled)
    # tsne_features_x = tsne_features[:, 0]
    # tsne_features_y = tsne_features[:, 1]

    # KMEANS
    cluster_num = 5
    kmeans = KMeans(n_clusters=cluster_num)
    kpred = kmeans.fit_predict(feature_scaled)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#FF5733', '#DAF7A6', '#C70039', '#900C3F', '#581845',
              '#1B1464', '#2C3E50', '#F4D03F', '#E74C3C', '#3498DB', '#A569BD', '#45B39D', '#922B21']
    markers = ['o', 'v', '*', '^', '8', 's', 'p', '*', 'h',
               'H', 'D', 'd', 'P', 'X', '+', '.', ',', '1', '2']

    cluster_states_dict = {
        device: [0 for i in range(cluster_num)] for device in devices}
    for i in range(len(features)):
        device = point_state_dict[i]
        cluster = kpred[i]
        cluster_states_dict[device][cluster] += 1
        # x = tsne_features_x[i]
        # y = tsne_features_y[i]
        # plt.scatter(x, y, c=colors[devices.index(
        #     device)], marker=markers[cluster], label=f'State: {device[:-6]}')

    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys(), loc="upper right")
    # plt.show()

    # ================== PIE CHART ==========================
    # Determine the layout for the subplots
    # num_devices = len(devices)
    # cols = 3  # 2 columns of pie charts, adjust if you prefer a different layout
    # # Ceiling division to get the number of rows
    # rows = -(-num_devices // cols)

    # # Create a new figure
    # fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6))

    # # Flatten axes object if it's 2D (i.e., more than one row and column)
    # if rows > 1 or cols > 1:
    #     axes = axes.ravel()

    # # Plot each pie chart
    # for i, (device, values) in enumerate(cluster_states_dict.items()):
    #     ax = axes[i]
    #     ax.pie(values, labels=values, colors=colors,
    #            autopct='%1.1f%%', startangle=140)
    #     ax.set_title(f"Data Distribution for {device[:-6]}")
    #     # Equal aspect ratio ensures pie is drawn as a circle.
    #     ax.axis('equal')

    # # Hide any remaining empty subplots
    # for j in range(i + 1, rows * cols):
    #     axes[j].axis('off')

    # # # Save the figure to a file
    # fig.savefig('pie_charts.png', bbox_inches='tight')

    # ================== CONFUSION MATRIX ==========================
    # Compute confusion matrix
    matrix = confusion_matrix(point_labels, kpred)
    matrix = matrix[:, :cluster_num]
    tick_labels = [device[:-6] for device in devices]

    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt='d',
                cmap='Blues', yticklabels=tick_labels)
    plt.xlabel('Predicted Cluster')
    plt.ylabel('True State')

    # Save the figure to a file
    plt.savefig('confusion_matrix_1.png', bbox_inches='tight')

    # If you still want to close the plot without displaying it
    plt.close()


def cal_clusters_center(device):
    states = app.database["iotstates"].find({"device": device})
    datas = app.database["iotdatas"].find({"device": device})
    data_dict = {}
    for state in states:
        state_idx = state["idx"]
        if state_idx != "-1" and state_idx != "-99" and state_idx not in data_dict:
            data_dict[state_idx] = []

    for data in datas:
        data_dict[data["state"]].append(data["data"])

    center_dict = {}
    for state, state_data in data_dict.items():
        center_point = calculate_centroid(state_data)
        center_dict[state] = center_point

    global centroids
    centroids = center_dict


def calculate_pairwise_distance(now_datas, other_center):
    total_distance = 0
    for d1 in now_datas:
        total_distance += calculate_distance(d1, other_center)

    avg = total_distance / len(now_datas)

    return avg
