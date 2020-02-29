from sklearn.decomposition import PCA
import pandas as pd


def pca_transformation(df, features_to_ignore, lower_limit=0.95):
    """

    :param df: dataframe
    :param features_to_ignore: list
    :param lower_limit: float from (0, 1)
    :return: (dataframe, list)
        vraca transformisane podatke i listu objasnjenih varijansi
    """
    data = df.drop(columns=features_to_ignore)
    pca = PCA(n_components=lower_limit).fit(data)
    transformed_data = pca.transform(data)

    return pd.DataFrame(transformed_data).join(df[features_to_ignore]), pca.explained_variance_ratio_
