from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from dotenv import dotenv_values
from pymongo import MongoClient
from urllib.parse import quote_plus
import certifi
import numpy as np
import time
import pickle
import multiprocessing
import power_data
from scipy import stats
from sklearn.preprocessing import StandardScaler
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

config = dotenv_values(".env")

username = quote_plus("haojian")
password = quote_plus('xwBVZV7fG8rjDKD')
cluster = 'cluster0.f8w36.mongodb.net'

uri = 'mongodb+srv://' + username + ':' + password + \
    '@' + cluster + '/?retryWrites=true&w=majority'

app = FastAPI()


@app.on_event("startup")
def startup_db_client():
    app.client = MongoClient(uri, tlsCAFile=certifi.where())
    app.database = app.client["iotdb"]
    print("Connected to the MongoDB database!")
    print(app.client)
    print(app.database)


@app.on_event("shutdown")
def shutdown_db_client():
    app.client.close()

# ========================================= Routes =========================================================


@app.get("/")
async def root():
    return {"message": "Here is the hardware end of IoT-Auditor!"}


@app.get("/get")
async def get_states_data():
    states_data = app.database["iotstates"].find()
    for state in states_data:
        print(state)
    return {"message": "hello"}


@app.get("/start")
async def start_sensing():
    device = "google home"  # TODO: change device to the board title!
    sensing(device)
    return {"message": "Sensing IoT device!"}


@app.get("/removeall")
async def remove_all():
    app.database["iotstates"].delete_many({})
    app.database["iotdatas"].delete_many({})
    return {"message": "Delete all data."}

# ========================================= Functions =========================================================


def calculate_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))  # Euclidean distance


def calculate_centroid(cluster):
    return np.mean(cluster, axis=0)


def get_closest_centroid(point, centroids):
    distances = [calculate_distance(point, centroid) for centroid in centroids]
    return np.argmin(distances), np.min(distances)


def create_state(state):
    new_state = app.database["iotstates"].insert_one(state)
    print(state)


def create_data(data):
    new_data = app.database["iotdatas"].insert_one(data)
    print(data)


def sensing(device):
    print("\n")
    start_time = time.time_ns()
    absolute_path = os.path.dirname(__file__)
    relative_path = "data"
    full_path = os.path.join(absolute_path, relative_path)

    states = ['power_on_unmuted', 'power_on_muted', 'change_volume_muted',
              'interact_muted', 'interact_unmuted', 'change_volume_unmuted']

    states_data = []
    data_points_info = []
    data_point_idx = 0
    state_clusters = []  # identified clusters
    centroids = []  # center point of clusters
    distance_threshold = 10000000  # TODO: threshold
    previous_data_cluster_idx = 0  # the state of previous data
    scaler_x = StandardScaler()

    new_state = True  # indicator just for testing

# ======================= Read data from data stream ========================================
    while(data_point_idx <= 30): #TODO: timing 
        q = multiprocessing.Queue()
        p2 = multiprocessing.Process(target=power_data.power_data, args=(q,))
        p2.start()
        p2.join()
        power = q.get()
        if len(power) > 0:
            features = [np.mean, np.var, lambda x: np.sqrt(np.mean(np.power(x, 2))), np.std, stats.median_abs_deviation, stats.skew, lambda x: stats.kurtosis(x, fisher=False), stats.iqr, lambda x: np.mean((x-np.mean(x))**2)]
            fea_power = [feature(power) for feature in features]
            fea = np.array(fea_power)
            fea = fea.reshape(1, fea.shape[0])
            # sample_transform = scaler_x.transform(fea)

            states_data.append(fea)

            # testing new box
            if data_point_idx % 5 == 0:
                new_state = True

            if len(centroids) == 0 or new_state:  # if this is the first point
                    state_clusters.append([fea])
                    centroids.append(fea)
                    cluster_idx = len(state_clusters) - 1
            else:
                closest_centroid_index, closest_distance = get_closest_centroid(fea, centroids)
                if closest_distance < distance_threshold:
                    closest_centroid_index = previous_data_cluster_idx # if distance less than threshold, then the state won't change?
                    state_clusters[closest_centroid_index].append(fea)
                    centroids[closest_centroid_index] = calculate_centroid(state_clusters[closest_centroid_index])
                    cluster_idx = closest_centroid_index
                else:
                    state_clusters.append([fea])
                    centroids.append(fea)
                    cluster_idx = len(state_clusters) - 1
            
            data_point_info = {
                "idx": str(data_point_idx),
                "state": str(cluster_idx),
                "data": fea.tolist(),
                "time": time.time_ns() - start_time,
                "device": device
            }
            data_point_idx += 1
            data_points_info.append(data_point_info)
            print(data_point_info)

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

# ======================= Read data from static files =======================================
    # for state in states:
    #     for file_num in range(1, 16):
    #         data_point_info = {}  # record the infomation of this data point
    #         power_file_name = 'power_features_' + state + str(file_num)
    #         power_data_path = os.path.join(full_path, power_file_name)
    #         # emalation_file_name = 'emalation_features_' + state + str(file_num)
    #         # emalation_data_path = os.path.join(full_path, emalation_file_name)

    #         with open(power_data_path, 'rb') as power_f:
    #             power_d = pickle.load(power_f)
    #         # with open(emalation_data_path, 'rb') as emanation_f:
    #         #     emanation_d = pickle.load(emanation_f)

    #         power_data = np.array(power_d)
    #         states_data.append(np.array(power_d))

    #         # Workflow:
    #         # 1. calculate distance between data point and clusters's center points
    #         # 2. if distance larger than threshold => create new cluster;
    #         # 3. if distance smaller than threshold => add data point to the nearest cluster => recalculate the center point of that cluster

    #         if len(centroids) == 0 or new_state:  # if this is the first point
    #             state_clusters.append([power_data])
    #             centroids.append(power_data)
    #             cluster_idx = len(state_clusters) - 1
    #         else:
    #             closest_centroid_index, closest_distance = get_closest_centroid(
    #                 power_data, centroids)
    #             if closest_distance < distance_threshold:
    #                 # if distance less than threshold, then the state won't change?
    #                 closest_centroid_index = previous_data_cluster_idx
    #                 state_clusters[closest_centroid_index].append(power_data)
    #                 centroids[closest_centroid_index] = calculate_centroid(
    #                     state_clusters[closest_centroid_index])
    #                 cluster_idx = closest_centroid_index
    #             else:
    #                 state_clusters.append([power_data])
    #                 centroids.append(power_data)
    #                 cluster_idx = len(state_clusters) - 1
    #                 new_state = True

    #         data_point_info = {
    #             "idx": str(data_point_idx),
    #             "state": str(cluster_idx),
    #             "data": power_data.tolist(),
    #             "time": time.time_ns() - start_time,
    #             "device": device
    #         }
    #         data_point_idx += 1
    #         data_points_info.append(data_point_info)

    #         if new_state:
    #             new_state_info = {
    #                 "time": time.time_ns() - start_time,
    #                 "device": device,
    #                 "idx": str(cluster_idx),
    #                 "prev_idx": str(previous_data_cluster_idx)
    #             }
    #             create_state(new_state_info)
    #             new_state = False
    #             previous_data_cluster_idx = cluster_idx  # record the current state

    #     new_state = True

    # states_data = np.array(states_data)

    # # # PCA
    # # pca = PCA(n_components=2)
    # # pca_transformed_data = pca.fit_transform(plot_data)

    # # TSNE
    # tsne = TSNE(n_components=2)
    # tsne_transformed_data = tsne.fit_transform(states_data)

    # for idx in range(len(data_points_info)):
    #     data_point = data_points_info[idx]
    #     reduction_result = tsne_transformed_data[idx].tolist()
    #     data_point["pos_x"] = reduction_result[0]
    #     data_point["pos_y"] = reduction_result[1]
    #     create_data(jsonable_encoder(data_point))
