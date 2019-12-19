from sklearn.cluster import KMeans, DBSCAN
import pandas as pd

def kmeans_clusters(df, min_k, max_k):
    """

    :param df: datafame
    :param min_k: int
        prirodan broj
    :param max_k: int
        prirodan broj
        mora da vazi max_k > min_k
    :return: dataframe
        konstruise novi dataframe koji za kolone ima labele instanci za odredjeno k
        broj koloni je max_k - min_k
    """

    clusters_labels = pd.DataFrame()
    for k in range(min_k, max_k + 1):
        est = KMeans(n_clusters=k).fit(df)
        clusters_labels[k] = est.labels_

    return clusters_labels
