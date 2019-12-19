import matplotlib.pyplot as plt
import preprocessing
from sklearn.manifold import TSNE

colors = ['red', 'green', 'blue', 'cyan', 'black', 'yellow', 'magenta', 'brown', 'plum', 'orange', 'darkcyan']


def scatter_plot_clusters(df):
    """

    :param df: dataframe
    :return: scatterplot
    """

    try:
        number_of_clusters = max(df['labels']) + 1
        for j in range(-1, number_of_clusters):
            if j == -1:
                label = 'noise'
            else:
                label = 'cluster %d' % j

            cluster = df.loc[df['labels'] == j]
            plt.scatter(x=cluster.iloc[:, 0],
                        y=cluster.iloc[:, 1],
                        color=colors[j],
                        label=label)

        plt.legend()
        plt.show()

    except Exception as e:
        print(e)


def barplot_explained_variance(explained_variance):
    """

    :param explained_variance: list
    :return: barplot
    """
    number_of_components = len(explained_variance)
    for ith_component in range(number_of_components):
        print("PC%d explained var: %f" % (ith_component + 1, explained_variance[ith_component]))

    plt.bar(["PC" + str(i+1) for i in range(number_of_components)], explained_variance)
    plt.xticks(rotation=90)
    plt.show()


def barplot_explained_variance_ratio(explained_variance_ratio):
    """

    :param explained_variance_ratio: list
    :return: barplot
    """
    barplot_explained_variance(explained_variance_ratio)


def knn_sorted_distance_plot(df, k):
    """

    :param df: dataframe
    :param k: int
    :return: scatter plot
        pronalazi k najblizih suseda za svaku tacku
    """
    kth_nearest_neighbors = preprocessing.kth_nearest_neighbor(df, k)
    kth_nearest_neighbors.sort(reverse=True)

    plt.scatter(x=range(len(kth_nearest_neighbors)),
                y=kth_nearest_neighbors,
                s=12)
    plt.show()


def tsne_plot(df):
    """

    :param df: dataframe
    :return: figure
        #TODO
        plotuje rezultat tsne-a za razlicite vrednosti perplexity-a
    """
    embed_data = TSNE(n_components=2,
                      perplexity=30).fit(df)
    plt.scatter(x=embed_data[:, 0],
                y=embed_data[:, 1])
    plt.show()
