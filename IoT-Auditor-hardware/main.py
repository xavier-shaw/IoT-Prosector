from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
from concurrent.futures.process import ProcessPoolExecutor
from typing import List
from dotenv import dotenv_values
from pymongo import MongoClient
from urllib.parse import quote_plus
import certifi
import numpy as np
import time
import pickle
import multiprocessing
import subprocess
import network_data
import power_data
import emanation_data
from scipy import stats
from sklearn.preprocessing import StandardScaler
import os
import math
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
import seaborn as sns
import matplotlib.pyplot as plt
import json
import threading
import serial
from paramiko import SSHClient
from contextlib import asynccontextmanager


config = dotenv_values(".env")
username = quote_plus(config["NAME"])
password = quote_plus(config["PASSWORD"])
cluster = config["CLUSTER"]
uri = 'mongodb+srv://' + username + ':' + password + \
    '@' + cluster + '/?retryWrites=true&w=majority'
pineapple_token = "eyJVc2VyIjoicm9vdCIsIkV4cGlyeSI6IjIwMjgtMDgtMjJUMDI6Mzc6NTUuMjk4NzgzMzAzWiIsIlNlcnZlcklkIjoiYTYyMTM3MzE3NTUyNDRlZSJ9.sbCLEXl3vWayXfd4zM2zgTthQnzEztZvWxNi_nejdvg="


# ====================================== GLOBAL VARIABLES ==================================================
ser = None
quit = False
listening = False
power_data_thread = None

max_currents = []
avg_currents = []
min_currents = []
times = []
start_time = 0

sample_num = 150
group_cnt = 10

tsne_data_points = []
tsne_data_labels = []
state_cluster_dict = {}
cluster_cnt = 0

classifier = None
labels = []
data_points = {}

SEMANTIC_CLUSTERING = 0
DATA_CLUSTERING = 1

q = multiprocessing.Queue()
finished_processes = []
processes = []

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_from_arduino():
    print("Start reading from Arduino...")
    global ser
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    global avg_currents, max_currents, min_currents, times, listening, quit
    while not quit:
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


power_data_thread = threading.Thread(target=read_from_arduino, daemon=True)
power_data_thread.start()


@app.on_event("startup")
async def startup_db_client():
    app.state.executor = ProcessPoolExecutor()
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
    global power_data_thread, ser, quit
    quit = True
    power_data_thread.join()
    ser.close()
    app.state.executor.shutdown()
    app.client.close()

# ========================================= Routes =========================================================


@app.get("/")
async def root():
    return {"message": "Here is the hardware end of IoT-Auditor!"}


@app.get("/getBoards")
async def get_boards():
    boards = app.database["boards"].find()
    titles = [board["title"] for board in boards]
    return {"message": str(titles)}


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
    app.database["iotdatas"].delete_many({})
    return {"message": "Delete all data."}


@app.get("/removeAllBoards")
async def remove_all_boards():
    app.database["boards"].delete_many({})
    return {"message": "delete all boards."}


@app.get("/startSensing")
async def start_sensing(device: str, idx: str, background_tasks: BackgroundTasks):
    global listening, start_time
    listening = True
    start_time = time.time()
    background_tasks.add_task(read_from_sm200, device, idx)

    return {"message": "Start Sensing for " + idx + "!"}


async def read_from_sm200(device, idx):
    global processes, finished_processes
    processes.append(idx)
    emanation_result = await run_in_process(emanation_data.emanation_data, idx)
    file_name = "/home/datasmith/Desktop/Iot-Auditor/IoT-Auditor/IoT-Auditor-hardware/fft_result/" + idx + ".pkl"
    with open(file_name, 'wb') as file:
        pickle.dump(emanation_result, file)
    finished_processes.append(idx)
    print("finish emanation data storing")


async def run_in_process(fn, *args):
    loop = asyncio.get_event_loop()
    # wait and return result
    return await loop.run_in_executor(app.state.executor, fn, *args)


@app.get("/stopSensing")
async def stop_sensing():
    global listening, max_currents, avg_currents, min_currents, times
    listening = False
    return {"message": "Stop Sensing!"}


@app.get("/storeData")
async def store(device, idx: str):
    store_power_data(device, idx, max_currents,
                     avg_currents, min_currents, times)
    clear_data()
    return {"message": "Store data"}


class DataModel(BaseModel):
    device: str
    nodes: List[dict]


@app.post("/waitForDataProcessing")
async def waitForProcessing(data: DataModel = Body(...)):
    global processes, finished_processes 
    while len(finished_processes) != len(processes):
        time.sleep(1)

    finished_processes = []
    processes = []

    data_model_hints, group_idx = process_data(data.device, data.nodes)

    resp = {
        "data_model_group_cnt": group_idx,
        "data_model_hints": data_model_hints
    }

    return JSONResponse(content=jsonable_encoder(resp))


def process_data(device, nodes):
    global tsne_data_points, tsne_data_labels, state_cluster_dict, cluster_cnt

    state_nodes = [n for n in nodes if n["type"] == "stateNode"]

    power_datas = []
    emanation_datas = []
    states_info = []

    power_datas, emanation_datas, states_info = get_all_raw_data(
        device, state_nodes)
    features, states_labels = data_processing(
        states_info, power_datas, emanation_datas)
    tsne_data_points = features
    tsne_data_labels = states_labels

    state_distribution_dict = cluster_states(features, states_labels)
    # data_model_hints: a dictionary that shows which data-oriented cluster the state is belonged to => hints[state_id] = cluster_id
    data_model_hints, group_idx = generate_collage_node(
        state_distribution_dict)

    state_cluster_dict = data_model_hints
    cluster_cnt = group_idx

    return data_model_hints, group_idx


@app.post("/collage")
async def action_collage(data: DataModel = Body(...)):
    actions, action_collage_dict = action_match(data.nodes)

    resp = {
        "action_group_count": len(actions),
        "action_collage_dict": action_collage_dict
    }

    return JSONResponse(content=jsonable_encoder(resp))


def get_state_group_info(nodes):
    state_group_info = {}
    parent_node_ids = []

    for node in nodes:
        node_id = node["id"]
        if node["parentNode"]:
            state_group_info[node_id] = node["parentNode"]
        else:
            state_group_info[node_id] = node_id
            parent_node_ids.append(node_id)
    
    return state_group_info, parent_node_ids


@app.post("/classification")
async def classfication(data: DataModel = Body(...)):
    global classifier, labels, data_points, tsne_data_points, state_cluster_dict, cluster_cnt
    # To show the cohesion level inside each group and coupling level between different groups
    # X: tsne_data_points (20 * num_of_states)
    # Y: group_labels   
    
    state_group_info, parent_node_ids = get_state_group_info(data.nodes)
    
    corr_matrix = np.zeros((cluster_cnt, len(parent_node_ids)))
    for state_id in tsne_data_labels:
        group_id = state_group_info[state_id]
        cluster_label = state_cluster_dict[state_id]
        corr_matrix[parent_node_ids.index(group_id), cluster_label] += 1

    row_sums = corr_matrix.sum(axis=1).reshape(-1, 1)
    corr_matrix /= row_sums
    corr_matrix = np.around(corr_matrix, 2)

    clusters = [("#" + i) for i in range(cluster_cnt)]
    groups = [node["data"]["label"] for node in data.nodes if node["id"] in parent_node_ids]
    resp = {
        "matrix": corr_matrix,
        "clusters": clusters,
        "groups": groups,
        "data_points": tsne_data_points,
        "data_labels": tsne_data_labels
    }
    return JSONResponse(content=jsonable_encoder(resp))


@app.get("/verify")
async def verify():
    global listening, max_currents, avg_currents, min_currents, times
    listening = False
    features = get_power_features(avg_currents, max_currents, min_currents)
    predict_state(features)
    clear_data()

    return {"message": "predict state"}


# ========================================= Functions =========================================================


def clear_data():
    global max_currents, avg_currents, min_currents, times
    max_currents = []
    avg_currents = []
    min_currents = []
    times = []


def create_data(data, type):
    app.database["iotdatas"].insert_one(data)
    print("store " + type + " data into database")


def store_power_data(device, idx, max_currents, avg_currents, min_currents, times):
    data = {
        "device": device,
        "idx": idx + "-power",
        "max_currents": max_currents,
        "avg_currents": avg_currents,
        "min_currents": min_currents,
        "times": times
    }
    create_data(jsonable_encoder(data), "power")


def action_match(nodes):
    actions = []      
    action_collage_dict = {}

    for node in nodes:
        action = node["data"]["action"]
        if action not in actions:
            actions.append(action)

        action_collage_dict[node["id"]] = actions.index(action)

    return actions, action_collage_dict


def get_all_raw_data(device, nodes):
    global state_group_info
    power_datas = []
    emanation_datas = []
    states_info = []

    for node in nodes:
        avg_currents, max_currents, min_currents, times = get_power_data(
            device, node["id"])
        emanation_data = get_emanation_data(device, node["id"])
        power_datas.append(avg_currents)
        emanation_datas.append(emanation_data)
        states_info.append(node["id"])
        state_group_info[node["id"]] = node["id"]

    # return labels, collected_nodes, X, Y, Y_true
    return power_datas, emanation_datas, states_info


def build_dataset(device, nodes):
    collected_nodes = []
    labels = []
    X = []
    Y = []
    Y_true = []

    order = ["semanticNode", "stateNode"]
    sorted_nodes = sorted(nodes, key=lambda x: order.index(x["type"]))

    # labels, collected_nodes, X, Y, Y_true = dfs_traverse_graph(
    #     device, sorted_nodes, sorted_nodes, labels, collected_nodes, X, Y, Y_true, independent=True)

    power_datas = []
    emanation_datas = []
    states_info = []
    groups_info = []
    labels, collected_nodes, power_datas, emanation_datas, states_info, groups_info = get_all_raw_data(
        device, sorted_nodes, sorted_nodes, labels, collected_nodes, power_datas, emanation_datas, states_info, groups_info, independent=True)

    # process data to get X, Y, Y_true_state, Y_true_group
    #

    X = np.array(X)
    Y = np.array(Y)

    return labels, collected_nodes, X, Y, Y_true


def get_power_features(avg_currents, max_currents, min_currents):
    group_size = 5
    power_features = []

    for i in range(len(avg_currents)):
        start_idx = i * group_size
        end_idx = (i + 1) * group_size
        if end_idx > len(avg_currents):
            break
        avgs = avg_currents[start_idx: end_idx]
        maxs = max_currents[start_idx: end_idx]
        mins = min_currents[start_idx: end_idx]
        # avg_features = [fea(avgs) for fea in [
        #     np.mean, np.std, lambda x: np.max(x) - np.min(x)]]
        # max_features = [fea(maxs) for fea in [np.mean, np.std]]
        # min_features = [fea(mins) for fea in [np.mean, np.std]]

        avg_features = [fea(avgs) for fea in [np.mean]]
        # max_features = [fea(maxs) for fea in [np.mean]]
        # min_features = [fea(mins) for fea in [np.mean]]

        feas = np.concatenate(
            (avg_features))

        power_features.append(feas)

    return power_features


def get_emanation_features(emanation_data, groups_cnt):
    group_size = math.floor(len(emanation_data) / groups_cnt)
    emanation_features = []

    for i in range(len(emanation_data)):
        start_idx = i * group_size
        end_idx = (i+1) * group_size
        if end_idx > len(emanation_data) or len(emanation_features) == groups_cnt:
            break
        fea = np.mean(emanation_data[start_idx: end_idx], axis=0)
        emanation_features.append(fea)

    return emanation_features


def get_emanation_data(device, state_idx):
    # data = app.database["iotdatas"].find_one(
    #     {"device": device, "idx": state_idx + "-emanation"})
    # emanation_data = data["emanation"]

    # file_name = "/home/datasmith/Desktop/Iot-Auditor/IoT-Auditor/IoT-Auditor-hardware/fft_result/" + state_idx + ".pkl"
    # with open(file_name, 'rb') as file:
    #     emanation_data = pickle.load(file)

    file_name = "/home/datasmith/Desktop/Iot-Auditor/IoT-Auditor/IoT-Auditor-hardware/fft_result/" + state_idx + ".pkl"
    emanation_result = []
    with open(file_name, 'rb') as file:
        power_result = pickle.load(file)
        for final_power in power_result:
            emanation_res = np.array([
                np.mean(final_power),
                np.median(final_power),
                np.std(final_power),
                np.var(final_power),
                np.average(final_power),
                np.sqrt(np.mean(final_power**2)),
                stats.median_abs_deviation(final_power),
                stats.skew(final_power),
                stats.kurtosis(final_power, fisher=False),
                stats.iqr(final_power),
                np.mean((final_power - np.mean(final_power))**2)
            ])
            emanation_result.append(emanation_res)
    return emanation_result


def get_power_data(device, state_idx):
    data = app.database["iotdatas"].find_one(
        {"device": device, "idx": state_idx + "-power"})
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


def generate_collage_node(distribution_dict, node_collage_dict):
    cluster_map = {}
    group_idx = 0
    for state, clusters in distribution_dict.items():
        cluster = int(np.argmax(clusters))
        if cluster not in cluster_map:
            cluster_map[cluster] = group_idx
            group_idx += 1
        node_collage_dict[state] = cluster_map[cluster]

    return node_collage_dict, group_idx


def cluster_states(X, Y):
    num_of_states = len(set(Y))

    # Silhouette Score
    clusters_nums = [i for i in range(2, num_of_states + 1)]
    sil = []
    for i in clusters_nums:  # Silhouette score is defined only for more than 1 cluster
        kmeans = KMeans(n_clusters=i, init='k-means++',
                        max_iter=300, n_init=10, random_state=42)
        kmeans.fit(X)
        silhouette_avg = silhouette_score(X, kmeans.labels_)
        sil.append(silhouette_avg)

    # plt.plot(range(type + 2, num_of_states + 1), sil)
    # plt.title('Silhouette Score Method')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Silhouette Score')

    # plt.savefig('silhouette_score.png', bbox_inches='tight')
    # plt.close()

    best_cluster_number = clusters_nums[np.argmax(sil)]
    kmeans = KMeans(n_clusters=best_cluster_number,
                    init='k-means++', max_iter=300, n_init=10, random_state=42)
    kpred = kmeans.fit_predict(X)

    distribution_dict = {l: [0 for i in range(
        best_cluster_number)] for l in range(num_of_states)}
    for i in range(len(Y)):
        true_label = Y[i]
        pred_label = kpred[i]
        distribution_dict[true_label][pred_label] += 1

    return distribution_dict


def build_semantic_dataset(node_collage_dict, ids, X, Y, new_X, new_Y):
    for index in range(len(Y)):
        node_id = ids[Y[index]]
        semantic_label = node_collage_dict[node_id]
        new_X.append(X[index])
        new_Y.append(semantic_label)

    return new_X, new_Y


def build_final_collage_node(node_collage_dict, final_distribution_dict, group_idx):
    cluster_map = {}
    final_collage_dict = {}
    for state, clusters in final_distribution_dict.items():
        cluster = int(np.argmax(clusters))
        if cluster not in cluster_map:
            cluster_map[cluster] = group_idx
            group_idx += 1
        for id, semantic_state in node_collage_dict.items():
            if state == semantic_state:
                final_collage_dict[id] = cluster_map[cluster]

    return final_collage_dict, group_idx
# ===========================================================================================================


def predict_state(features):
    global classifier, labels
    pred = []
    for feature in features:
        state = classifier.predict(feature)
        pred.append(state)

    print(pred)


# ===========================================================================================================
def data_processing(states, raw_power_data, raw_emanation_data):
    max_len = 11  # largest length of emanation vector
    # list with (number of states x  10), 10 means each state will have 10 data features
    powers = []
    state_labels = []

    for state_idx in range(len(states)):
        # state power data: average values of current => size: (1 * 200)
        # state emanation data: data from the .32cf file => size: (500, k)
        state_power_data = raw_power_data[state_idx]
        state_emanation = raw_emanation_data[state_idx]
        i = 0
        num = 20
        power = []  # one iot state will have 10 examples which is averaged over 20 power data points
        while (i < 200):
            powers.append(state_power_data[i:i+num])
            state_labels.append(states[state_idx])  # the state of the data
            i = i+num
    # converting powers to be array
    powers_fea = np.array(powers)

    # interpolating the emanations
    # 10 indicates that one iot state has 10 examples
    emanations_fea = np.zeros((len(states)*10, max_len))
    interp_index = 0
    for state_idx in range(len(states)):  # interpolating emanation vectors
        state_emanation = raw_emanation_data[state_idx]
        emanation_interp = np.zeros((500, max_len))
        num = 0
        for i in range(len(state_emanation)):
            max_emanation = max(state_emanation[i])
            min_emanation = min(state_emanation[i])
            emanation_interp[i, :] = (
                state_emanation[i]-min_emanation)/(max_emanation-min_emanation)

        # taking average for the interpolated emanations, since there are 10 power data examples, the emanation data examples should be 10.
        # so we average over 50 emanation data points.
        num_examples = 50
        j = 0
        while (j < 500):
            emanations_fea[interp_index, :] = np.mean(
                emanation_interp[j:j+num_examples, :], axis=0)
            interp_index = interp_index + 1
            j = j + j+num_examples

    # conducting  tsne to show the 2d scattering plot
    print("power: ", powers_fea.shape)
    print("emanation: ", emanations_fea.shape)
    conc_feas = np.hstack((powers_fea, emanations_fea))
    embedded_feas = TSNE(n_components=2, learning_rate='auto',
                         init='random', perplexity=5).fit_transform(conc_feas)

    return embedded_feas.tolist(), state_labels
