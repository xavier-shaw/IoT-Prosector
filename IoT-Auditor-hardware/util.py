import pickle
import pandas as pd
import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data_points_info = {}
clusters = [[] for i in range(7)]
scaled_clusters = [[] for i in range(7)]

absolute_path = os.path.dirname(__file__)
relative_path = "data"
directory_path = os.path.join(absolute_path, relative_path)

exps = ['muted_100', 'unmuted_100', 'unmute_interaction_100', 'muted_interaction_100', 'volume_change_unmuted_100', 'volume_change_muted_100', 'power_off_100']

feats = []
for i, exp in enumerate(exps):
    file_name = 'features_' + exp + '.pkl'
    full_path = os.path.join(directory_path, file_name)
    with open(full_path, 'rb') as f:
        fe = pd.read_pickle(f)
        fe['exp'] = exp
        fe['explab'] = i 
        feats.append(fe)
comb_feats = pd.concat(feats, axis=0).reset_index(drop=True)
print(comb_feats)

features = comb_feats.iloc[:,:-2]
y = comb_feats.iloc[:,-1]

sc = StandardScaler()
features_scaled = sc.fit_transform(features)

# TSNE
tsne = TSNE(n_components = 2)
tsne_features = tsne.fit_transform(features_scaled)

tsne_features_x = tsne_features[:, 0]
tsne_features_y = tsne_features[:, 1]

# KMEANS
kmeans = KMeans(n_clusters=7)
kpred = kmeans.fit_predict(tsne_features)

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#FF5733', '#DAF7A6', '#C70039', '#900C3F', '#581845', '#1B1464', '#2C3E50', '#F4D03F', '#E74C3C', '#3498DB', '#A569BD', '#45B39D', '#922B21']
markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X', '+', '.', ',', '1', '2']

for idx in range(700):
    data_point_info = {
        "idx": idx,
        "features": features.iloc[idx],
        "scaled_features": features_scaled[idx],
        "state": y.iloc[idx],
        "tsne_data": tsne_features[idx][:],
        "kmeans_cluster": kpred[idx]
    }

    clusters[kpred[idx]].append(features.iloc[idx])
    scaled_clusters[kpred[idx]].append(features_scaled[idx])
    data_points_info[idx] = data_point_info
    plt.scatter(x=data_point_info["tsne_data"][0], y=data_point_info["tsne_data"][1], c=colors[data_point_info['kmeans_cluster']], marker=markers[data_point_info['state']])
plt.show()

def calculate_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))  # Euclidean distance


def calculate_centroid(cluster):
    return np.mean(cluster, axis=0)


def get_closest_centroid(point, centroids):
    distances = [calculate_distance(point, centroid) for centroid in centroids]
    return np.argmin(distances), np.min(distances)

centers = []
radiuses = []
scaled_centers = []
scaled_radiuses = []

for i in range(7):
    cluster = clusters[i]
    center = calculate_centroid(cluster)
    total_distance = 0
    for point in cluster:
        distance = calculate_distance(point, center)
        total_distance += distance
    radius = total_distance / len(cluster)
    centers.append(center)
    radiuses.append(radius)

    scaled_cluster = scaled_clusters[i]
    scaled_center = calculate_centroid(scaled_cluster)
    total_distance_scaled = 0
    for point in scaled_cluster:
        distance = calculate_distance(point, scaled_center)
        total_distance_scaled += distance
    scaled_radius = total_distance_scaled / len(scaled_cluster)
    scaled_centers.append(scaled_center)
    scaled_radiuses.append(scaled_radius)

    print("cluster " + str(i) + ": center - " + str(center) + ", radius - " + str(radius))
    print("scaled cluster " + str(i) + ": scaled center - " + str(scaled_center) + ", scaled radius - " + str(scaled_radius))
    
avg_radius = np.mean(radiuses)
print(avg_radius)

avg_scaled_radius = np.mean(scaled_radiuses)
print(avg_scaled_radius)

# inside_cnt = 0
# scaled_inside_cnt = 0
# for i in range(7):
#     cluster = clusters[i]
#     center = centers[i]
#     for point in cluster:
#         distance = calculate_distance(point, center)
#         if distance < avg_radius:
