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

classifier = None
labels = []
data_points = {}

SEMANTIC_CLUSTERING = 0
DATA_CLUSTERING = 1

q = multiprocessing.Queue()
state_infos = []
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
    os.environ['IDX'] = idx
    background_tasks.add_task(read_from_sm200, device, idx)

    return {"message": "Start Sensing for " + idx + "!"}


@app.get("/stopSensing")
async def stop_sensing():
    global listening, max_currents, avg_currents, min_currents, times
    listening = False
    return {"message": "Stop Sensing!"}

async def read_from_sm200(device, idx):
    global q, state_infos, processes
    # p = multiprocessing.Process(
    #     target=emanation_data.emanation_data, args=(q, idx))
    # os.environ['IDX']= idx
    # state_infos.append((device, idx))
    # processes.append(p)
    # p.start()
    processes.append(idx)
    emanation_result = await run_in_process(emanation_data.emanation_data, idx)
    state_infos.append(idx)
    # data = {
    #     "device": device,
    #     "idx": idx + "-emanation",
    #     "emanation": emanation_result
    # }
    # create_data(jsonable_encoder(data), "emanation")
    file_name = "/home/datasmith/Desktop/Iot-Auditor/IoT-Auditor/IoT-Auditor-hardware/fft_result/" + idx + ".pkl"
    with open(file_name, 'wb') as file:
        pickle.dump(emanation_result, file)
    print("finish emanation data storing")


async def run_in_process(fn, *args):
    loop = asyncio.get_event_loop()
    # wait and return result
    return await loop.run_in_executor(app.state.executor, fn, *args)


@app.get("/waitForDataProcessing")
async def waitForProcessing():
    global q, state_infos, processes
    # for process in processes:
    #     process.join()

    # for device, idx in state_infos:
    #     emanation_data = q.get()
    #     data = {
    #         "device": device,
    #         "idx": idx + "-emanation",
    #         "emanation": emanation_data.tolist()
    #     }
    #     create_data(jsonable_encoder(data), "emanation")
    print("states: ", len(state_infos))
    while len(state_infos) != len(processes):
        time.sleep(1)

    state_infos = []
    processes = []

    return {"message": "Finish Processing!"}


@app.get("/storeData")
async def store(device, idx: str):
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
    # 2. run clustering algorithm inside each group => semantic group states
    # 3. based on semantic result, run clustering algorithm again => combined group state
    # 4. we have a heirarchical graph

    semantic_group_idx = 0
    semantic_node_collage_dict = {}

    actions, action_node_dict, action_collage_dict = action_match(data.nodes)
    new_X = []
    new_Y = []
    for action, nodes in action_node_dict.items():
        states, ids, X, Y, Y_true = build_dataset(data.device, nodes)
        semantic_distribution_dict = cluster_states(X, Y, SEMANTIC_CLUSTERING)
        semantic_node_collage_dict, semantic_group_idx = generate_collage_node(
            semantic_distribution_dict, ids, semantic_group_idx, semantic_node_collage_dict)
        new_X, new_Y = build_semantic_dataset(
            semantic_node_collage_dict, ids, X, Y, new_X, new_Y)

    combined_group_idx = 0
    combined_distribution_dict = cluster_states(new_X, new_Y, DATA_CLUSTERING)
    combined_node_collage_dict, combined_group_idx = build_final_collage_node(
        semantic_node_collage_dict, combined_distribution_dict, combined_group_idx)

    resp = {
        "action_group_count": len(actions),
        "action_collage_dict": action_collage_dict,
        "semantic_group_cnt": semantic_group_idx,
        "semantic_collage_dict": semantic_node_collage_dict,
        "combined_group_cnt": combined_group_idx,
        "combined_collage_dict": combined_node_collage_dict
    }

    print("action", action_collage_dict)
    print("sementic-data", semantic_node_collage_dict)
    print("data", combined_node_collage_dict)
    return JSONResponse(content=jsonable_encoder(resp))


@app.post("/classification")
async def classfication(data: DataModel = Body(...)):
    global classifier, labels, data_points

    states, ids, X, Y, Y_true = build_dataset(data.device, data.nodes)

    # classification model
    clf, cms, acc = classify(states, X, Y)
    classifier = clf
    labels = states

    # tsne dta points
    data_points_info = []
    if len(data_points) > 0:
        data_points_info = data_points
    else:
        # tsne
        # scaler = StandardScaler()
        # X_scaled = scaler.fit_transform(X)
        print("re - tsne")
        tsne = TSNE(n_components=2, perplexity=5)
        X_scaled = tsne.fit_transform(X)
        for i in range(len(X_scaled)):
            data_points_info.append({
                "x": float(X_scaled[i][0]),
                "y": float(X_scaled[i][1]),
                "label": Y_true[i]
            })

    resp = {
        "accuracy": round(acc, 3),
        "confusionMatrix": cms,
        "states": states,
        "dataPoints": data_points_info
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


@app.get("/emanationTest")
async def emanation_test():
    device = "test"
    nodes = app.database["boards"].find_one({"title": device})["chart"]["nodes"]
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
    labels, collected_nodes, power_datas, emanation_datas, states_info, groups_info = dfs_traverse_graph(
        device, sorted_nodes, sorted_nodes, labels, collected_nodes, power_datas, emanation_datas, states_info, groups_info, independent=True)
    print(len(states_info))
    print(len(power_datas))
    print(len(emanation_datas))
    data_processing(states_info, power_datas, emanation_datas)

    return {"message": "test emanation process"}
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
    action_node_dict = {}
    action_collage_dict = {}

    for node in nodes:
        action = node["data"]["action"]
        if action in actions:
            action_node_dict[action].append(node)
        else:
            actions.append(action)
            action_node_dict[action] = [node]

        action_collage_dict[node["id"]] = actions.index(action)

    return actions, action_node_dict, action_collage_dict


def dfs_traverse_graph(device, nodes, target_nodes, labels, collected_nodes, power_datas, emanation_datas, states_info, groups_info, independent=False):
    for node in target_nodes:
        if node["id"] not in collected_nodes:
            collected_nodes.append(node["id"])
            if independent:
                labels.append(node["data"]["label"])

            if "children" in node["data"]:
                children = node["data"]["children"]
                if len(children) > 0:
                    children_nodes = [n for n in nodes if n["id"] in children]
                    # labels, collected_nodes, X, Y, Y_true = dfs_traverse_graph(
                    #     device, nodes, children_nodes, labels, collected_nodes, X, Y, Y_true)
                    labels, collected_nodes, power_datas, emanation_datas, states_info, groups_info = dfs_traverse_graph(
                        device, nodes, children_nodes, labels, collected_nodes, power_datas, emanation_datas, states_info, groups_info)
            else:
                avg_currents, max_currents, min_currents, times = get_power_data(
                    device, node["id"])
                emanation_data = get_emanation_data(device, node["id"])
                power_datas.append(avg_currents)
                emanation_datas.append(emanation_data)
                # states_info.append(node["id"])
                states_info.append(node["data"]["label"])
                if "parentNode" in node["data"]:
                    groups_info.append(node["data"]["parentNode"])
                else:
                    groups_info.append(node["id"])
                # power_features = get_power_features(
                #     avg_currents, max_currents, min_currents)
                # emanation_data = get_emanation_data(device, node["id"])
                # emanation_features = get_emanation_features(
                #     emanation_data, len(power_features))
                # features = np.hstack((power_features, emanation_features))
                # features = power_features
                # for feature in features:
                #     X.append(feature)
                #     Y.append(len(labels) - 1)
                #     Y_true.append(node["id"])

    # return labels, collected_nodes, X, Y, Y_true
    return labels, collected_nodes, power_datas, emanation_datas, states_info, groups_info


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
    labels, collected_nodes, power_datas, emanation_datas, states_info, groups_info = dfs_traverse_graph(
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
    print("start: ", state_idx)
    emanation_result = emanation_data.recalculate_emanation_data(state_idx)
    print("finish: ", state_idx)
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


def generate_collage_node(distribution_dict, ids, group_idx, node_collage_dict):
    cluster_map = {}
    for state, clusters in distribution_dict.items():
        cluster = int(np.argmax(clusters))
        if cluster not in cluster_map:
            cluster_map[cluster] = group_idx
            group_idx += 1
        node_collage_dict[ids[state]] = cluster_map[cluster]

    return node_collage_dict, group_idx


def cluster_states(X, Y, type):
    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X)
    num_of_states = len(set(Y))

    # Silhouette Score
    clusters_nums = [i for i in range(type + 2, num_of_states + 1)]
    sil = []
    for i in clusters_nums:  # Silhouette score is defined only for more than 1 cluster
        kmeans = KMeans(n_clusters=i, init='k-means++',
                        max_iter=300, n_init=10, random_state=42)
        kmeans.fit(scaled_X)
        silhouette_avg = silhouette_score(scaled_X, kmeans.labels_)
        sil.append(silhouette_avg)

    # plt.plot(range(type + 2, num_of_states + 1), sil)
    # plt.title('Silhouette Score Method')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Silhouette Score')

    # plt.savefig('silhouette_score.png', bbox_inches='tight')
    # plt.close()

    best_cluster_number = 1
    if len(sil) > 0:
        best_cluster_number = clusters_nums[np.argmax(sil)]
    kmeans = KMeans(n_clusters=best_cluster_number,
                    init='k-means++', max_iter=300, n_init=10, random_state=42)
    kpred = kmeans.fit_predict(scaled_X)

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
    max_len = 0  # largest length of emanation vector
    powers = [] # list with (number of states x  10), 10 means each state will have 10 data features 
    for state_idx in range(len(states)):
        # state power data: average values of current => size: (1 * 200)
        # state emanation data: data from the .32cf file => size: (500, k)
        state_power_data = raw_power_data[state_idx]
        print("power data size: ", len(state_power_data))
        state_emanation = raw_emanation_data[state_idx]
        i=0
        num = 20
        power = []# one iot state will have 10 examples which is averaged over 20 power data points
        while(i<200):
           powers.append(state_power_data[i:i+num])
           i=i+num
        # powers.append(power)
        # get the largest length of the emantion vector
        # print(len(state_emanation))
        for i in range(len(state_emanation)):
            state_emanation_example = state_emanation[i]
            max_len = max(max_len, len(state_emanation_example))
    # converting powers to be array
    powers_fea = np.array(powers)
    
    # interpolating the emanations 
    emanations_fea = np.zeros((len(states)*10, max_len)) # 10 indicates that one iot state has 10 examples
    interp_index = 0
    for state_idx in range(len(states)): # interpolating emanation vectors
        state_emanation = raw_emanation_data[state_idx]
        emanation_interp = np.zeros((500, max_len))
        num = 0
        for i in range(len(state_emanation)):
            # num = num + 1
            x = np.linspace(0,1,len(state_emanation[i]))
            xvals = np.linspace(0,1,max_len)
            emanation_interp[i,:]= np.interp(xvals, x, state_emanation[i])
        # taking average for the interpolated emanations, since there are 10 power data examples, the emanation data examples should be 10.
        # so we average over 50 emanation data points.
        num_examples = 50
        j=0
        while(j<500):
            emanations_fea[interp_index,:] = np.mean(emanation_interp[j:j+num_examples,:], axis=0)
            interp_index = interp_index + 1
            j = j + j+num_examples

    # conducting  tsne to show the 2d scattering plot
    print("power: ", powers_fea.shape)
    print("emanation: ", emanations_fea.shape)
    conc_feas = np.hstack((powers_fea, emanations_fea))
    embedded_feas = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=5).fit_transform(conc_feas)
    
    # scatter plot
    customColors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
'#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8',
'#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']
    uniq_labels = set()
    for i in range(embedded_feas.shape[0]):
        state = states[int(i / 10)]
        plt.scatter(embedded_feas[i][0], embedded_feas[i][1], color=customColors[int(i / 10)], label=state if state not in uniq_labels else "")
        uniq_labels.add(state)
    plt.legend(loc='best')
    # plt.savefig('tsne.png', bbox_inches='tight')
    plt.show()