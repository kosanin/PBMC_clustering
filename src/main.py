import numpy as np
import visualization
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import preprocessing
import decomposition
import clustering
from clustering import kmeans_clusters
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

def main():


    df = pd.read_csv("data/iris.csv")
    df = df.drop(columns=['variety'])
    # n x n matrica rastojanja
    W = pairwise_distances(df.values, metric='euclidean')
    #
    print(W)



if __name__ == '__main__':
    main()
