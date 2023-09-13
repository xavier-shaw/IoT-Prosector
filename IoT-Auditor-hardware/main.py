from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
from concurrent.futures.process import ProcessPoolExecutor
from typing import List, Dict
from dotenv import dotenv_values
from pymongo import MongoClient
from urllib.parse import quote_plus
import certifi
import numpy as np
import pandas as pd
import time
import pickle
from collections import Counter
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
from sklearn.model_selection import KFold, train_test_split
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
finish_write = True
emanation_sweep = 400

sample_num = 150
group_cnt = 10

tsne_data_points_train = []
tsne_data_labels_train = []
tsne_data_points_test = []
tsne_data_labels_test = []
state_cluster_dict = {}
cluster_cnt = 0
classifier = None
state_group_dict = {}

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
    global processes, finished_processes, finish_write
    processes.append(idx)
    finish_write = False
    finish_write = await run_in_process(emanation_data.emanation_data, idx)
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
    global max_currents, avg_currents, min_currents, times
    store_power_data(device, idx, max_currents,
                     avg_currents, min_currents, times)
    clear_data()
    return {"message": "Store data"}


class DataModel(BaseModel):
    device: str
    nodes: List[dict]


class ProcessedDataModel(BaseModel):
    tsne_data_labels: List[str]
    tsne_data_points: List[List[float]]
    state_cluster_dict: Dict[str, List[int]]
    cluster_cnt: int


@app.post("/loadProcessedData")
async def load_processed_data(data: ProcessedDataModel = Body(...)):
    global tsne_data_points_train, tsne_data_labels_train, tsne_data_points_test, tsne_data_labels_test, state_cluster_dict, cluster_cnt
    tsne_data_points_train = np.array(data.tsne_data_points_train)
    tsne_data_labels_train = np.array(data.tsne_data_labels_train)
    tsne_data_points_test = np.array(data.tsne_data_points_test)
    tsne_data_labels_test = np.array(data.tsne_data_labels_test)
    state_cluster_dict = data.state_cluster_dict
    cluster_cnt = data.cluster_cnt

    return {"message": "data are loaded"}


@app.post("/waitForDataProcessing")
async def wait_for_data_processing(data: DataModel = Body(...)):
    global processes, finished_processes
    while len(finished_processes) != len(processes):
        time.sleep(1)

    finished_processes = []
    processes = []

    resp = process_data(data.nodes)

    return JSONResponse(jsonable_encoder(resp))


def process_data(nodes):
    global tsne_data_points_train, tsne_data_labels_train, tsne_data_points_test, tsne_data_labels_test, state_cluster_dict, cluster_cnt

    state_nodes = [n for n in nodes if n["type"] == "stateNode"]

    power_datas = []
    emanation_datas = []
    states_info = []

    power_datas, emanation_datas, states_info = get_all_raw_data(state_nodes)
    features_train, features_test, states_labels_train, states_labels_test = data_processing(
        states_info, power_datas, emanation_datas)
    tsne_data_points_train = features_train
    tsne_data_labels_train = states_labels_train
    tsne_data_points_test = features_test
    tsne_data_labels_test = states_labels_test

    state_distribution_dict, best_cluster_number = cluster_states(
        features_train, states_labels_train)

    state_cluster_dict = state_distribution_dict
    cluster_cnt = best_cluster_number

    resp = {
        "tsne_data_labels_train": tsne_data_labels_train.tolist(),
        "tsne_data_points_train": tsne_data_points_train.tolist(),
        "tsne_data_labels_test": tsne_data_labels_test.tolist(),
        "tsne_data_points_test": tsne_data_points_test.tolist(),
        "state_cluster_dict": state_cluster_dict,
        "cluster_cnt": cluster_cnt
    }

    return resp


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
        if "parentNode" in node and node["parentNode"]:
            state_group_info[node_id] = node["parentNode"]
        else:
            state_group_info[node_id] = node_id
            parent_node_ids.append(node_id)

    return state_group_info, parent_node_ids


@app.post("/classification")
async def classfication(data: DataModel = Body(...)):
    global tsne_data_points_train, state_cluster_dict, cluster_cnt
    # To show the cohesion level inside each group and coupling level between different groups
    # X: tsne_data_points (20 * num_of_states)
    # Y: group_labels

    state_group_info, parent_node_ids = get_state_group_info(data.nodes)

    corr_matrix = np.zeros((cluster_cnt, len(parent_node_ids)))
    for state_id, group_id in state_group_info.items():
        if state_id in state_cluster_dict:
            cluster_distribution = state_cluster_dict[state_id]
            corr_matrix[:, parent_node_ids.index(
                group_id)] += cluster_distribution

    col_sums = corr_matrix.sum(axis=0)
    corr_matrix /= col_sums
    corr_matrix = np.around(corr_matrix, 2)

    clusters = [("Cluster " + str(i)) for i in range(cluster_cnt)]
    groups = [node["data"]["label"]
              for node in data.nodes if node["id"] in parent_node_ids]
    resp = {
        "matrix": corr_matrix.tolist(),
        "clusters": clusters,
        "groups": groups,
        "data_points": tsne_data_points_train,
        "data_labels": tsne_data_labels_train
    }

    return JSONResponse(content=jsonable_encoder(resp))


@app.post("/train")
async def train_model(data: DataModel = Body(...)):
    global tsne_data_points_train, tsne_data_labels_train, state_group_dict, classifier
    nodes = data.nodes

    parentNodes = [n for n in nodes if (
        "parentNode" not in n) or (not n["parentNode"])]
    node_group_dict = {}
    group_idx = 0

    for parentNode in parentNodes:
        state_group_dict[group_idx] = parentNode["id"]
        if "children" in parentNode["data"]:
            for child in parentNode["data"]["children"]:
                node_group_dict[child] = group_idx
        else:
            node_group_dict[parentNode["id"]] = group_idx
        group_idx += 1

    X = []
    Y = []

    for i in range(len(tsne_data_points_train)):
        data_point = tsne_data_points_train[i]
        data_label = tsne_data_labels_train[i]
        group_label = node_group_dict[data_label]

        X.append(data_point)
        Y.append(group_label)

    clf = DecisionTreeClassifier()
    clf.fit(X, Y)

    classifier = clf

    return {"message": "classifier model trained"}


@app.get("/predict")
async def predict():
    global classifier, state_group_dict, tsne_data_points_test, tsne_data_labels_test
    
    # Combine X_test and y_test into a DataFrame
    df = pd.DataFrame(tsne_data_points_test, columns=["X1", "X2"])
    df["Label"] = tsne_data_labels_test

    # Group by label and calculate the average for each group
    averages = df.groupby("Label").mean()

    # Convert the resulting DataFrame back to a NumPy array (if needed)
    average_X = averages.values
    average_Y = averages.index 

    predict_results = classifier.predict(average_X)

    def map_to_dict(value):
        return state_group_dict[value]

    predict_results = np.vectorize(map_to_dict)(predict_results)
    print("result", predict_results)

    clear_data()

    resp = {
        "predict_data_points": average_X.tolist(),
        "predict_states": predict_results.tolist(),
        "original_labels": average_Y.tolist()
    }
    return JSONResponse(resp)


@app.get('/verify')
async def verify(device: str, predict: str, correct: str):
    file_name = "/home/datasmith/Desktop/Iot-Auditor/IoT-Auditor/IoT-Auditor-hardware/verification_result/" + device + ".txt"
    with open(file_name, 'a') as file:
        # Write content to the file
        content = predict + " & " + correct + "\n"
        file.write(content)

    return {"message": "verification result submitted."}
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


def get_all_raw_data(nodes):
    power_datas = []
    emanation_datas = []
    states_info = []

    for node in nodes:
        avg_currents, max_currents, min_currents, times = get_power_data(
            node["id"])
        emanation_data = get_emanation_data(node["id"])
        power_datas.append(avg_currents)
        emanation_datas.append(emanation_data)
        states_info.append(node["id"])

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


def get_emanation_data(state_idx, wait=False):
    # data = app.database["iotdatas"].find_one(
    #     {"device": device, "idx": state_idx + "-emanation"})
    # emanation_data = data["emanation"]

    # file_name = "/home/datasmith/Desktop/Iot-Auditor/IoT-Auditor/IoT-Auditor-hardware/fft_result/" + state_idx + ".pkl"
    # with open(file_name, 'rb') as file:
    #     emanation_data = pickle.load(file)
    global finish_write
    file_name = "/home/datasmith/Desktop/Iot-Auditor/IoT-Auditor/IoT-Auditor-hardware/fft_result/" + state_idx + ".pkl"
    while not os.path.exists(file_name):
        time.sleep(5)

    emanation_result = []
    power_results = []
    with open(file_name, 'rb') as file:
        try:
            while True:
                power_results.append(pickle.load(file))
        except EOFError:
            pass

    for power_result in power_results:
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


def get_power_data(state_idx):
    data = app.database["iotdatas"].find_one(
        {"idx": state_idx + "-power"})
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


def generate_collage_node(distribution_dict):
    cluster_map = {}
    node_collage_dict = {}
    group_idx = 0
    for state, clusters in distribution_dict.items():
        cluster = int(np.argmax(clusters))
        if cluster not in cluster_map:
            cluster_map[cluster] = group_idx
            group_idx += 1
        node_collage_dict[state] = cluster_map[cluster]

    return node_collage_dict, group_idx


def cluster_states(X, Y):
    true_labels = set(Y)

    # Silhouette Score
    clusters_nums = [i for i in range(2, len(true_labels) + 1)]
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

    distribution_dict = {label: [0 for i in range(
        best_cluster_number)] for label in true_labels}
    for i in range(len(Y)):
        true_label = Y[i]
        pred_label = kpred[i]
        distribution_dict[true_label][pred_label] += 1

    return distribution_dict, best_cluster_number


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


def data_processing(states, raw_power_data, raw_emanation_data):
    global data_points, emanation_sweep
    max_len = 11  # largest length of emanation vector

    # list with (number of states x  10), 10 means each state will have 10 data features
    powers = []
    state_labels = []

    for state_idx in range(len(states)):
        # state power data: average values of current => size: (1 * 200)
        # state emanation data: data from the .32cf file => size: (500, k)
        state_power_data = raw_power_data[state_idx]
        i = 0
        num = 8
        power = []  # one iot state will have 10 examples which is averaged over 20 power data points
        while (i < 200):
            powers.append(state_power_data[i:i+num])
            state_labels.append(states[state_idx])  # the state of the data
            i = i+num

    # converting powers to be array
    powers_fea = np.array(powers)
    state_labels = np.array(state_labels)

    # interpolating the emanations
    # 10 indicates that one iot state has 10 examples
    emanations_fea = np.zeros((len(states)*10, max_len))
    interp_index = 0
    for state_idx in range(len(states)):  # interpolating emanation vectors
        state_emanation = raw_emanation_data[state_idx]
        emanation_interp = np.zeros((emanation_sweep, max_len))
        num = 0
        for i in range(len(state_emanation)):
            max_emanation = max(state_emanation[i])
            min_emanation = min(state_emanation[i])
            emanation_interp[i, :] = (
                state_emanation[i]-min_emanation)/(max_emanation-min_emanation)

        # taking average for the interpolated emanations, since there are 10 power data examples, the emanation data examples should be 10.
        # so we average over 50 emanation data points.
        num_examples = int(emanation_sweep / 25)
        j = 0
        while (j < emanation_sweep):
            emanations_fea[interp_index, :] = np.mean(
                emanation_interp[j:j+num_examples, :], axis=0)
            interp_index = interp_index + 1
            j = j + num_examples

    # conducting  tsne to show the 2d scattering plot
    print("power: ", powers_fea.shape)
    print("emanation: ", emanations_fea.shape)
    conc_feas = np.hstack((powers_fea, emanations_fea))

    embedded_feas = TSNE(n_components=2, learning_rate='auto',
                         init='random', perplexity=5).fit_transform(conc_feas)
    feas_train, feas_test, labels_train, labels_test = train_test_split(
        embedded_feas, state_labels, test_size=0.2, stratify=state_labels, random_state=42)
    return feas_train, feas_test, labels_train, labels_test