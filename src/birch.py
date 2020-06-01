import pandas as pd 
import os, sys
from sklearn.cluster import Birch
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances, davies_bouldin_score, silhouette_score

def scatter_plot_clusters(df):
    plt.figure(figsize=(8, 7))
    number_of_clusters = max(df['labels']) + 1
    for j in range(-1, number_of_clusters):
        if j == -1:
            label = 'noise'
        else:
            label = 'cluster %d' % j
            
        cluster = df.loc[df['labels'] == j]
        plt.scatter(x=cluster.iloc[:, 0], y=cluster.iloc[:, 1], label=label, alpha=0.75, edgecolor='black')
    plt.legend()
    plt.show()
def extract_cluster(df, cluster_label, labels):
    return df.loc[df[labels] == cluster_label]
def get_file_distr_per_cluster(df):
    file_distr = df['cell_id'].value_counts()
    return file_distr
def plot_all_file_dists(columns):
    for column in columns:
        tmp = cells_and_labels[['cell_id', column]]
        tmp_file_dist = file_dist.copy()

        number_of_clusters = cells_and_labels[column].nunique()

        for cluster_label in range(number_of_clusters):
            cluster = extract_cluster(cells_and_labels, cluster_label, column)
            clust_dist = get_file_distr_per_cluster(cluster)
            clust_dist = clust_dist.rename("cluster: %d" % cluster_label)
            tmp_file_dist = tmp_file_dist.join(clust_dist, on="filename", how="left")

        tmp_file_dist.plot.bar(subplots=True, color=colors[1:], figsize=(6,11), x=0)
        plt.xticks(rotation=45)
        plt.savefig("proba.png")

df = pd.read_csv("group1_NMF.csv")
tsne_data = pd.read_csv("group1_TSNE_embedding.csv")
cells_and_labels = pd.DataFrame(df['cell_id'], columns=['cell_id'])
df = df.drop(columns=['cell_id'])

db = Birch(
	branching_factor=30,
	threshold = 0.2,
	n_clusters=6,
	).fit(df)
tsne_data['labels'] = db.labels_
cells_and_labels['labels']=db.labels_
scatter_plot_clusters(tsne_data)
#print(cells_and_labels)
file_dist = pd.DataFrame({"filename" : cells_and_labels['cell_id'].apply(lambda x: x[:10]).unique()})
#print(file_dist)
cells_and_labels['cell_id'] = cells_and_labels['cell_id'].apply(lambda cell_id : cell_id[:10])
#print(cells_and_labels)

silhouette_score_value_euclidean = silhouette_score(df, db.labels_, metric='euclidean')
silhouette_score_value_cosine = silhouette_score(df, db.labels_, metric='cosine')
print("silhouette_score_value_cosine", silhouette_score_value_cosine)
print("silhouette_score_value_euclidean", silhouette_score_value_euclidean)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plot_all_file_dists(cells_and_labels)
