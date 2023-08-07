import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import stats
from scipy import signal
import plotly.express as px
import plotly.graph_objects as go

exps = ['muted_100', 'unmuted_100', 'unmute_interaction_100', 'muted_interaction_100', 'volume_change_unmuted_100', 'volume_change_muted_100', 'power_off_100']

feats = []
for i, exp in enumerate(exps):
    fname = exp + '/features_' + exp + '.pkl'
    with open(fname, 'rb') as f:
        fe = pickle.load(f)
        fe['exp'] = exp
        fe['explab'] = i 
        feats.append(fe)
comb_feats = pd.concat(feats, axis=0).reset_index(drop=True)
comb_feats

from sklearn.preprocessing import StandardScaler
X = comb_feats.iloc[:,:-2]
y = comb_feats.iloc[:,-1]

sc = StandardScaler()
X_scaled = sc.fit_transform(X)

from sklearn.decomposition import PCA
components = 2
pca = PCA(n_components = components)
pca.fit(X_scaled)

arr = pca.transform(X_scaled)

pc1 = arr[:, 0]
pc2 = arr[:, 1]

fig = px.scatter(x=pc1, y=pc2, color=y, title="PCA")
fig.show()

from sklearn.manifold import TSNE
perplexity=40
tsne = TSNE(n_components = components, perplexity=perplexity, init='pca')
arr = tsne.fit_transform(X_scaled)

tsne1 = arr[:, 0]
tsne2 = arr[:, 1]
fig = px.scatter(x=tsne1, y=tsne2, color=y, title='Perplexity: ' + str(perplexity) + ", Init: PCA, Features: Unscaled")
fig.show()


# # Clustering

# from sklearn.cluster import DBSCAN
# clustering = DBSCAN(eps=2.2, min_samples=5).fit(arr)
# c_labels = clustering.labels_
# c_labels_matched = np.empty_like(c_labels)

# # For each cluster label...
# for c in np.unique(c_labels):

#     # ...find and assign the best-matching truth label
#     match_nums = [np.sum((c_labels==c)*(y==t)) for t in np.unique(y)]
#     print(match_nums, c)
#     c_labels_matched[c_labels==c] = np.unique(y)[np.argmax(match_nums)]

# print()
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y, c_labels_matched)
# # Plot confusion matrix
# ax = plt.axes()
# ax.set_facecolor("white")
# plt.imshow(cm,interpolation='none',cmap='Blues')
# for (i, j), z in np.ndenumerate(cm):
#     plt.text(j, i, z, ha='center', va='center')
# plt.xlabel("kmeans label")
# plt.ylabel("truth label")

# #plt.savefig("Confusion_Matrix.png")
# plt.show()


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=7)
kpred = kmeans.fit_predict(arr)

k_labels = kmeans.labels_  # Get cluster labels
k_labels_matched = np.empty_like(k_labels)

# For each cluster label...
for k in np.unique(k_labels):

    # ...find and assign the best-matching truth label
    match_nums = [np.sum((k_labels==k)*(y==t)) for t in np.unique(y)]
    k_labels_matched[k_labels==k] = np.unique(y)[np.argmax(match_nums)]

fig = px.scatter(x=tsne1, y=tsne2, color=y, title="Ground Truth")
fig.show()
fig = px.scatter(x=tsne1, y=tsne2, color=k_labels_matched, title="KMeans Prediction")
fig.show()
    
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, k_labels_matched)
# Plot confusion matrix
ax = plt.axes()
ax.set_facecolor("white")
plt.imshow(cm,interpolation='none',cmap='Blues')
for (i, j), z in np.ndenumerate(cm):
    plt.text(j, i, z, ha='center', va='center')
plt.xlabel("kmeans label")
plt.ylabel("truth label")

#plt.savefig("Confusion_Matrix")
plt.show()

# from sklearn.manifold import MDS
# embed = MDS(n_components=2)
# arr = embed.fit_transform(X_scaled)
# mds1 = arr[:, 0]
# mds2 = arr[:, 1]
# fig = px.scatter(x=mds1, y=mds2, color=y)
# fig.show()


# # Extract Features out of PSDs

exp = 'power_off_100'

exps = ['muted_100', 'unmuted_100', 'unmute_interaction_100', 'muted_interaction_100', 'volume_change_unmuted_100', 'volume_change_muted_100', 'power_off_100']

step = 200 # mHz
combined_dfs = []
x_ss = []
for CF in range(200, 799, step):
    # File should be your manually selected Xs with getDataTips
    df_x = pd.read_csv('./' + exp + '/X_' + exp + '_' + str(CF) + '.csv', header=None)
    df_x = df_x.rename(columns={0:'freq (MHz)', 1:'Power (dB/Hz)'})
    df_x['freq (MHz)'] = df_x['freq (MHz)'].str.lstrip('X ').astype(float).round(6)
    df_x['Power (dB/Hz)'] = df_x['Power (dB/Hz)'].str.lstrip('Y ').astype(float)
    df_x = df_x.sort_values('freq (MHz)')
    df_x['ExperimentNum'] = 0
    x_s = df_x['freq (MHz)']
    x_ss.append(x_s)
    print(x_s.shape)
    
    # Reads in all of the files
    dfs = []
    for i in range(0, 100):
        df = pd.read_csv('./' + exp + '/Capture' + str(i) 
                         + '_t_100mS_CF_' + str(CF) + 'MHz_BW_200MHz.csv', header=None)
        df = df.rename(columns={0:'freq (MHz)', 1:'Power (dB/Hz)'})
        df['freq (MHz)'] = (df['freq (MHz)'].astype(float)).round(6)
        df['Power (dB/Hz)'] = df['Power (dB/Hz)'].astype(float)
        df['ExperimentNum'] = i
        dfs.append(df[df['freq (MHz)'].isin(x_s)])
    combined = pd.concat(dfs, axis=0).reset_index(drop=True)
    combined['freq (MHz)'] = combined['freq (MHz)'] * step + CF
    combined_dfs.append(combined)
    print(combined.shape)
    #combined.to_csv(str(CF) + "GH_" + exp + ".csv")
combined_df = pd.concat(combined_dfs, axis=0).reset_index(drop=True)
combined_df

combined_df.to_pickle(exp + '/combined_' + exp + '.pkl')


# combined_dfs = []
# for i in range(200, 799, 200):
#     df = pd.read_csv(str(i) + "GH_" + exp + ".csv")
#     combined_dfs.append(df)
# combined = pd.concat(combined_dfs, axis=0).reset_index(drop=True).drop(columns={'Unnamed: 0'})
# combined.to_picked('combined_' + exp + '.pkl')


# # Testing FindPeaks

# i = 16
# exp = 'unmute_interaction'
# CF = 600

# df = pd.read_csv('./' + exp + '/Capture' + str(i) 
#                          + '_t_100mS_CF_' + str(CF) + 'MHz_BW_200MHz.csv', header=None)
# df = df.rename(columns={0:'freq (MHz)', 1:'Power (dB/Hz)'})
# df['freq (MHz)'] = (df['freq (MHz)'].astype(float)).round(6)
# df['Power (dB/Hz)'] = df['Power (dB/Hz)'].astype(float)

# peaks = signal.find_peaks(df['Power (dB/Hz)'], distance=14, prominence=0.4, threshold=0.01)

# found_peaks = df.iloc[peaks[0],:]

# fig1 = px.line(df, x='freq (MHz)', y='Power (dB/Hz)')
# fig2 = px.scatter(found_peaks, x='freq (MHz)', y='Power (dB/Hz)')
# fig3 = go.Figure(data=fig1.data + fig2.data)
# fig3.show()

# 200CF distance=8, prominence=0.1, threshold=0.05
# 400CF distance=16, prominence=0.1, threshold=0.03
# 600CF distance=14, prominence=0.4, threshold=0.01


# # Automatic PCA with Find Peaks

# step = 200
# states = ['on_muted', 'on', 'unmute_interaction', 'mute_interaction', 'volume_change']

# feats = []
# for state_name in states:
#     state = []
#     for CF in range(200, 799, 200):
#         caps = []
#         for i in range(0, 20):
#             df = pd.read_csv('./' + state_name + '/Capture' + str(i) 
#                                  + '_t_100mS_CF_' + str(CF) + 'MHz_BW_200MHz.csv', header=None)
#             df = df.rename(columns={0:'freq (MHz)', 1:'Power (dB/Hz)'})
#             df['freq (MHz)'] = (df['freq (MHz)'].astype(float)).round(6) * step + CF
#             df['Power (dB/Hz)'] = df['Power (dB/Hz)'].astype(float)
#             df['expnum'] = i
#             if CF == 200:
#                 peak_ix = signal.find_peaks(df['Power (dB/Hz)'], distance=8, prominence=0.1, threshold=0.05)[0]
#             elif CF == 400:
#                 peak_ix = signal.find_peaks(df['Power (dB/Hz)'], distance=16, prominence=0.1, threshold=0.03)[0]
#             elif CF == 600:
#                 peak_ix = signal.find_peaks(df['Power (dB/Hz)'], distance=14, prominence=0.4, threshold=0.01)[0]
#             else:
#                 peak_ix = signal.find_peaks(df['Power (dB/Hz)'], distance=5, prominence=0.2)[0]
#             peaks = df.iloc[peak_ix, :]
#             caps.append(peaks)
#         CF_peaks = pd.concat(caps, axis=0).reset_index(drop=True)
#         state.append(CF_peaks)
#     com_state = pd.concat(state, axis=0).reset_index(drop=True)
#     features = com_state.groupby("expnum")['Power (dB/Hz)'].aggregate(func=[lambda x: np.mean(abs(x)), 'var', 
#                                                               lambda x: np.sqrt(np.mean(x**2)), 'std', 
#                                                               stats.median_abs_deviation, stats.skew, 
#                                                               lambda x: stats.kurtosis(x, fisher=False), stats.iqr, 
#                                                               lambda x: np.mean((x - np.mean(x))**2)])
#     features = features.rename(columns={'<lambda_0>':'mav', '<lambda_1>':'RMSE', '<lambda_2>':'kurtosis', '<lambda_3>':'MSE'})
#     features['state'] = state_name
#     feats.append(features)
# comb_feats = pd.concat(feats, axis=0).reset_index(drop=True)
# comb_feats


# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# X = comb_feats.iloc[:,:-1]
# y = comb_feats.iloc[:,-1]

# sc = StandardScaler()
# X_scaled = sc.fit_transform(X)


# components = 2
# pca = PCA(n_components = components)
# arr = pca.fit_transform(X_scaled)
# pc1 = arr[:, 0]
# pc2 = arr[:, 1]
# fig = px.scatter(x=pc1, y=pc2, color=y)
# fig.show()


# print("Cumulative Variances (Percentage):")
# print(pca.explained_variance_ratio_.cumsum() * 100)
# print()

