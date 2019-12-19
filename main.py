import numpy as np
import visualization
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import preprocessing
import decomposition
from clustering import kmeans_clusters
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

def main():


    df = pd.read_csv("data/iris.csv")
    df = df.drop(columns=['variety'])
    #df = df.drop(columns=['Unnamed: 0', 'cell_id'])
    #df = preprocessing.drop_low_variance_columns(df)
    #df = preprocessing.log_transformation(df, [])
    #pca_data, explained_variance = decomposition.pca_transformation(df, [])
    #visualization.barplot_explained_variance(explained_variance)
    #df.to_csv("nozero_low_var_pca.csv")
    visualization.knn_sorted_distance_plot(df,9)
    #
    # est = DBSCAN(eps=284, min_samples=9).fit(df)
    # embed_data = TSNE(n_components=2,
    #                   perplexity=30, n_iter=5000).fit_transform(df)
    # x = pd.DataFrame(embed_data)
    # x['labels'] = est.labels_
    # visualization.scatter_plot_clusters(x)
    #x['variety'] = y
    #print(x)
    #x.to_csv("temp.csv")


if __name__ == '__main__':
    main()
