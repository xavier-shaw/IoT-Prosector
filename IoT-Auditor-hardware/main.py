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
import emanation_data
from scipy import stats
from sklearn.preprocessing import StandardScaler
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

config = dotenv_values(".env")

username = quote_plus(config["NAME"])
password = quote_plus(config["PASSWORD"])
cluster = config["CLUSTER"]


uri = 'mongodb+srv://' + username + ':' + password + \
    '@' + cluster + '/?retryWrites=true&w=majority'

app = FastAPI()


@app.on_event("startup")
def startup_db_client():
    app.client = MongoClient(uri, tlsCAFile=certifi.where())
    app.database = app.client[config["DB_NAME"]]
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


@app.get("/get_data/")
async def get_data(device: str):
    data = app.database["iotdatas"].find({"device": device})
    # print(data)
    center, radius = compute_data_features(data)
    return {"message": str(center) + " and " + str(radius)}
    # return {"center": center, "radius": radius}


@app.get("/draw")
async def draw_tsne():
    draw()
    return {"message": "draw tsne chart"}


@app.get("/start/")
async def start_sensing(device: str):
    sensing(device)
    return {"message": "Sensing IoT device at real time: " + device}


@app.get("/startlocal/")
async def start_local_sensing(device: str):
    local_sensing(device)
    return {"message": "Sensing IoT device from local file: " + device}


@app.get("/remove/")
async def remove(device: str):
    app.database["iotstates"].delete_many({"device": device})
    app.database["iotdatas"].delete_many({"device": device})
    return {"message": "Delete all data of " + device}


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
    app.database["iotstates"].insert_one(state)
    print("==================== NEW STATE =====================")
    print(state)
    print("==================== NEW STATE =====================")


def create_data(data):
    app.database["iotdatas"].insert_one(data)
    print(data)


def sensing(device):
    print("\n")
    start_time = time.time_ns()
    absolute_path = os.path.dirname(__file__)
    relative_path = "data"
    full_path = os.path.join(absolute_path, relative_path)

    states = ['power_on_unmuted', 'power_on_muted', 'change_volume_muted',
              'interact_muted', 'interact_unmuted', 'change_volume_unmuted']

    points_data = []
    data_points_info = {}
    data_point_idx = 0
    state_clusters = []  # identified clusters
    centroids = []  # center point of clusters
    # TODO: threshold for outlier
    # unscaled: 0.4981052741155125  scaled: 0.6203916378805225
    distance_threshold = 0.4981052741155125
    previous_data_cluster_idx = -1  # the state of previous data
    # TODO: threshold for new cluster (times of continous outlier)
    count_threshold = 1000
    outlier_buffer = []  # a buffer array for potential new cluster
    outlier_buffer_idx = []  # an array records the potential outliers' idx
    scaler_x = StandardScaler()  # the scaler for normalization
    new_state = True  # indicator for creating a new state

    boring_time = 0  # indicator for the time since last new state was created
    boring_threshold = 100  # the threshold for stable states

# ======================= Read data from data stream ========================================
    while (len(state_clusters) < 7 and boring_time <= boring_threshold):
        boring_time += 1
        q = multiprocessing.Queue()
        p2 = multiprocessing.Process(target=power_data.power_data, args=(q,))
        p3 = multiprocessing.Process(
            target=emanation_data.emanation_data, args=(q,))

        p2.start()
        p3.start()

        p2.join()
        p3.join()

        power = q.get()
        emanation = q.get()

        if len(power) > 0:
            features = [np.mean, np.var, lambda x: np.sqrt(np.mean(np.power(x, 2))), np.std, stats.median_abs_deviation, stats.skew, lambda x: stats.kurtosis(
                x, fisher=False), stats.iqr, lambda x: np.mean((x-np.mean(x))**2)]
            fea_power = [feature(power) for feature in features]
            fea_emanation = [feature(emanation) for feature in features]
            fea = np.array(fea_power + fea_emanation)
            fea = fea.reshape(1, fea.shape[0])
            print("features: ", fea)
            points_data.append(fea)

            # TESTING: creating a new state every 5 data points
            # if data_point_idx % 5 == 0:
            #     new_state = True

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
                    print("now center")
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
                    print("Next state is: " + belonged_cluster_idx)
                # larger than threshold
                else:
                    print("larger than threshold")
                    print("outlier buffer count: " + len(outlier_buffer))
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
                        cluster_idx = -1  # indicate this data point is an outlier
                        # add the idx to the buffer so that its state can be updated later
                        outlier_buffer_idx.append(data_point_idx)

            data_point_info = {
                "idx": str(data_point_idx),
                "state": str(cluster_idx),
                "data": fea.tolist(),
                "time": time.time_ns() - start_time,
                "device": device
            }
            # TODO: For testing
            create_data(jsonable_encoder(data_point_info))

            data_points_info[data_point_idx] = data_point_info
            data_point_idx += 1
            print(data_point_info)

            if new_state:
                boring_time = 0  # reset the boring time
                new_state_info = {
                    "time": time.time_ns() - start_time,
                    "device": device,
                    "idx": str(cluster_idx),
                    "prev_idx": str(previous_data_cluster_idx)
                }
                create_state(new_state_info)
                new_state = False
                previous_data_cluster_idx = cluster_idx  # record the current state

    # Retrospective Workflow:
    # 1. now we collect all data points and states, we can first normalize the data
    # 2. then we use TSNE to reduct the data vectors into 2-dimensional
    # 3. we store the data points with its features into the database
    transformed_points_data = scaler_x.fit_transform(points_data)
    tsne = TSNE(n_components=2)
    tsne_points_data = tsne.fit_transform(transformed_points_data)
    for idx in range(len(data_points_info)):
        data_point = data_points_info[idx]
        data_point["transformed_data"] = transformed_points_data[idx].tolist()
        data_point["tsne_data"] = tsne_points_data[idx].tolist()
        create_data(jsonable_encoder(data_point))


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
    for d in data:
        features.append(d["data"])

    center_point = calculate_centroid(features)
    total_distance = 0
    for data_point in features:
        distance = calculate_distance(data_point, center_point)
        total_distance += distance

    radius = total_distance / len(features)

    return center_point, radius


def draw():
    devices = ["power_on_muted", "power_on_unmuted", "unmuted_volume_change",
               "muted_volume_change", "muted_interaction", "unmuted_interaction"]
    features = []
    point_state_dict = {}
    idx = 0
    for device in devices:
        data = app.database["iotdatas"].find({"device": device})
        print(data)
        for d in data:
            point_state_dict[idx] = device
            idx += 1
            features.append(d["data"][0])
    print(features)
    scaler = StandardScaler()
    feature_scaled = scaler.fit_transform(features)

    tsne = TSNE(n_components=2, perplexity=40, init="pca")
    tsne_features = tsne.fit_transform(feature_scaled)
    tsne_features_x = tsne_features[:, 0]
    tsne_features_y = tsne_features[:, 1]

    # KMEANS
    kmeans = KMeans(n_clusters=len(devices))
    kpred = kmeans.fit_predict(tsne_features)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#FF5733', '#DAF7A6', '#C70039', '#900C3F', '#581845',
              '#1B1464', '#2C3E50', '#F4D03F', '#E74C3C', '#3498DB', '#A569BD', '#45B39D', '#922B21']
    markers = ['o', 'v', '*', '^', '8', 's', 'p', '*', 'h',
               'H', 'D', 'd', 'P', 'X', '+', '.', ',', '1', '2']

    for i in range(len(features)):
        device = point_state_dict[i]
        cluster = kpred[i]
        x = tsne_features_x[i]
        y = tsne_features_y[i]
        plt.scatter(x, y, c=colors[devices.index(
            device)], marker=markers[cluster])
    plt.show()
