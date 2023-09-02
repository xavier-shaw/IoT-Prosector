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
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
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
quit = False
listening = False
data_thread = None

max_currents = []
avg_currents = []
min_currents = []
times = []
start_time = 0

sample_num = 150
group_cnt = 10

classifier = None
labels = []


def read_from_arduino():
    print("Start reading from Arduino...")
    global ser
    ser = serial.Serial('/dev/tty.usbmodem21101', 9600, timeout=1)
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
    global data_thread, ser, quit
    quit = True
    data_thread.join()
    ser.close()
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
async def start_sensing():
    global listening, start_time
    listening = True
    start_time = time.time()
    return {"message": "Start Recording!"}


@app.get("/stopSensing")
async def stop_sensing(idx: str, device: str):
    global listening, max_currents, avg_currents, min_currents, times
    listening = False
    return {"message": "Stop Recording!"}


@app.get("/storeData")
async def store(idx: str, device: str):
    store_power_data(device, idx, max_currents,
                     avg_currents, min_currents, times)
    clear_data()
    return {"message": "Store data"}


class DataModel(BaseModel):
    device: str
    nodes: List[dict]


@app.post("/collage")
async def collage(data: DataModel = Body(...)):
    # 1. split nodes to different groups by action
    # 2. run clustering algorithm inside each group
    group_idx = 0
    node_collage_dict = {}

    actions, action_node_dict = action_match(data.nodes)
    for action, nodes in action_node_dict.items():
        states, ids, X, Y = build_dataset(data.device, nodes)
        distribution_dict = cluster_states(X, Y)
        node_collage_dict, group_idx = generate_collage_node(distribution_dict, ids, group_idx, node_collage_dict)

    resp = {
        "group_cnt": group_idx,
        "node_collage_dict": node_collage_dict
    }
    return JSONResponse(content=jsonable_encoder(resp))


@app.post("/classification")
async def classfication(data: DataModel = Body(...)):
    global classifier, labels
    states, ids, X, Y = build_dataset(data.device, data.nodes)
    clf, cms, acc = classify(states, X, Y)
    classifier = clf
    labels = states

    resp = {
        "accuracy": round(acc, 3),
        "confusionMatrix": cms,
        "states": states
    }
    return JSONResponse(content=jsonable_encoder(resp))


@app.get("/verify")
async def verify():
    global listening, max_currents, avg_currents, min_currents, times
    listening = False
    features = get_features(avg_currents, max_currents, min_currents)
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


def create_data(data):
    app.database["iotdatas"].insert_one(data)
    print("store data into database")


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

def action_match(nodes):
    actions = []
    action_node_dict = {}
    for node in nodes:
        action = node["data"]["action"]
        if action in action_node_dict:
            action_node_dict[action].append(node)
        else:
            actions.append(action)
            action_node_dict[action] = [node]
    
    return actions, action_node_dict


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
                avg_currents, max_currents, min_currents, times = get_data(
                    device, child)
                features = get_features(
                    avg_currents, max_currents, min_currents)
                for feature in features:
                    dataset_X.append(feature)
                    dataset_Y.append(len(labels) - 1)

    for state_node in state_nodes:
        if state_node["id"] not in collected_node:
            collected_node.append(state_node["id"])
            labels.append(state_node["data"]["label"])
            avg_currents, max_currents, min_currents, times = get_data(
                device, state_node["id"])
            features = get_features(avg_currents, max_currents, min_currents)
            for feature in features:
                dataset_X.append(feature)
                dataset_Y.append(len(labels) - 1)

    dataset_X = np.array(dataset_X)
    dataset_Y = np.array(dataset_Y)
    return labels, collected_node, dataset_X, dataset_Y


def get_features(avg_currents, max_currents, min_currents):
    group_size = 5
    features = []

    for i in range(len(avg_currents)):
        start_idx = i * group_size
        end_idx = (i + 1) * group_size
        if end_idx > len(avg_currents):
            break
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
        print(y_pred)
        print(cm)
        cms += cm

        acc = accuracy_score(y_test, y_pred)
        acc_scores.append(acc)

    avg_accuracy = np.mean(acc_scores)
    avg_cm = []
    for row in cms:
        row_sum = np.sum(row)
        avg_cm.append([format(x / row_sum, ".2f") for x in row])
    print("cms:", cms)
    print("avg_cm", avg_cm)
    return clf, avg_cm, avg_accuracy


def generate_collage_node(distribution_dict, ids, group_idx, node_collage_dict):
    cluster_map = {}
    for state, clusters in distribution_dict.items():
        cluster = int(np.argmax(clusters))
        if cluster not in cluster_map:
            group_idx += 1
            cluster_map[cluster] = group_idx
        node_collage_dict[ids[state]] = cluster_map[cluster]

    return node_collage_dict, group_idx


def cluster_states(X, Y):
    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X)
    num_of_states = len(set(Y))

    # Silhouette Score
    clusters_nums = [i for i in range(2, num_of_states + 1)]
    sil = []
    for i in clusters_nums:  # Silhouette score is defined only for more than 1 cluster
        kmeans = KMeans(n_clusters=i, init='k-means++',
                        max_iter=300, n_init=10, random_state=0)
        kmeans.fit(scaled_X)
        silhouette_avg = silhouette_score(scaled_X, kmeans.labels_)
        sil.append(silhouette_avg)

    best_cluster_number = 1
    if len(sil) > 0:
        best_cluster_number = clusters_nums[np.argmax(sil)]
    kmeans = KMeans(n_clusters=best_cluster_number)
    kpred = kmeans.fit_predict(scaled_X)

    distribution_dict = {l: [0 for i in range(
        best_cluster_number)] for l in range(num_of_states)}
    for i in range(len(Y)):
        true_label = Y[i]
        pred_label = kpred[i]
        distribution_dict[true_label][pred_label] += 1

    return distribution_dict
    # plt.plot(range(2, len(states)), sil)
    # plt.title('Silhouette Score Method')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Silhouette Score')

    # plt.savefig('silhouette_score.png', bbox_inches='tight')
    # plt.close()
    # # # ===========================  DBSCAN ==================================
    # from sklearn.neighbors import NearestNeighbors
    # k_value = 3
    # # Compute the nearest neighbors
    # nn = NearestNeighbors(n_neighbors=k_value).fit(scaled_X)
    # distances, _ = nn.kneighbors(scaled_X)

    # # Sort distances by the distance to the kth nearest neighbor
    # sorted_distances = np.sort(distances[:, -1])
    # plt.plot(sorted_distances)
    # plt.ylabel('kth Nearest Neighbor Distance')
    # plt.xlabel('Points Sorted by Distance')
    # plt.savefig("test.png")

    # db = DBSCAN(eps=0.5, min_samples=k_value).fit(scaled_X)
    # labels = db.labels_
    # print(labels)
    # print(Y)
    # # Number of clusters in labels, ignoring noise if present.
    # unique_labels = set(labels)
    # n_clusters_ = len(unique_labels) - (1 if -1 in labels else 0)

# ===========================================================================================================


def predict_state(features):
    global classifier, labels
    pred = []
    for feature in features:
        state = classifier.predict(feature)
        pred.append(state)

    print(pred)
