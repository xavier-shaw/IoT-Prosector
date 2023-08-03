import numpy as np
import time
import pickle
import multiprocessing
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def calculate_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2)) # Euclidean distance

def calculate_centroid(cluster):
    return np.mean(cluster, axis=0)

def get_closest_centroid(point, centroids):
    distances = [calculate_distance(point, centroid) for centroid in centroids]
    return np.argmin(distances), np.min(distances)
# =============================================================================
print("\n")
absolute_path = os.path.dirname(__file__)
relative_path = "data"
full_path = os.path.join(absolute_path, relative_path)

states = ['power_on_unmuted', 'power_on_muted', 'change_volume_muted',
          'interact_muted', 'interact_unmuted', 'change_volume_unmuted']

states_data = []
data_points_info = []
data_point_idx = 0
state_clusters = [] # identified clusters
centroids = [] # center point of clusters
distance_threshold = 10000000 # TODO: threshold
new_state = True # indicator just for testing
previous_data_cluster_idx = 0 # the state of previous data 

# TODO: read data from state's file => read data from sensing data stream 
for state in states:
    for file_num in range(1, 16):
        data_point_info = {} # record the infomation of this data point
        power_file_name = 'power_features_' + state + str(file_num)
        power_data_path = os.path.join(full_path, power_file_name)
        # emalation_file_name = 'emalation_features_' + state + str(file_num)
        # emalation_data_path = os.path.join(full_path, emalation_file_name)

        with open(power_data_path, 'rb') as power_f:
            power_d = pickle.load(power_f)
        # with open(emalation_data_path, 'rb') as emanation_f:
        #     emanation_d = pickle.load(emanation_f)

        power_data = np.array(power_d)
        states_data.append(np.array(power_d))
        
        # Workflow:
        # 1. calculate distance between data point and clusters's center points
        # 2. if distance larger than threshold => create new cluster; 
        # 3. if distance smaller than threshold => add data point to the nearest cluster => recalculate the center point of that cluster
        
        if len(centroids) == 0 or new_state:  # if this is the first point
            state_clusters.append([power_data])
            centroids.append(power_data)
            cluster_idx = len(state_clusters) - 1
            new_state = False
        else:
            closest_centroid_index, closest_distance = get_closest_centroid(power_data, centroids)
            if closest_distance < distance_threshold:
                closest_centroid_index = previous_data_cluster_idx # if distance less than threshold, then the state won't change?
                state_clusters[closest_centroid_index].append(power_data)
                centroids[closest_centroid_index] = calculate_centroid(state_clusters[closest_centroid_index])
                cluster_idx = closest_centroid_index
            else:
                state_clusters.append([power_data])
                centroids.append(power_data)
                cluster_idx = len(state_clusters) - 1
        
        previous_data_cluster_idx = cluster_idx # record the current state
        data_point_info = {
            "id": data_point_idx,
            "cluster_idx": cluster_idx,
            "data": power_data,
            "state": state
        }
        data_point_idx += 1
        data_points_info.append(data_point_info)
    
    new_state = True
# =============================================================================

states_data = np.array(states_data)

# # PCA
# pca = PCA(n_components=2)
# pca_transformed_data = pca.fit_transform(plot_data)

# TSNE
tsne = TSNE(n_components=2)
tsne_transformed_data = tsne.fit_transform(states_data)

# plot data
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#FF5733', '#DAF7A6', '#C70039', '#900C3F', '#581845', '#1B1464', '#2C3E50', '#F4D03F', '#E74C3C', '#3498DB', '#A569BD', '#45B39D', '#922B21']
markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X', '+', '.', ',', '1', '2']

for idx in range(len(data_points_info)):
    data_point = data_points_info[idx]
    reduction_result = tsne_transformed_data[idx]
    data_point["position_x"] = reduction_result[0]
    data_point["position_y"] = reduction_result[1]
    plt.scatter(data_point["position_x"], data_point["position_y"], c=colors[states.index(data_point["state"])], marker=markers[data_point["cluster_idx"]], label=f'{data_point["state"]} - {data_point["cluster_idx"]}')

# make plot legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc="lower right")

plt.show()
print(data_points_info)