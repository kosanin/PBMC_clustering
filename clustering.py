import sklearn
import pandas as pd
import numpy as np

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
        est = sklearn.cluster.KMeans(n_clusters=k).fit(df)
        clusters_labels[k] = est.labels_

    return clusters_labels


def dbscan_clusters(df, eps_min, eps_max, number_of_eps, min_samples):
    """

    :param df: dataframe
    :param eps_min: number
    :param eps_max: number
        require eps_max > eps_min
    :param number_of_eps: number
    :param min_samples: number
    :return: dataframe
    """
    cluster_labels = pd.DataFrame()
    eps_values = np.linspace(eps_min, eps_max, number_of_eps, endpoint=True)
    for eps in eps_values:
        est = sklearn.cluster.DBSCAN(eps, min_samples).fit(df)
        cluster_labels['eps %f min_samples %d' % (eps, min_samples)] = est.labels_

    return cluster_labels