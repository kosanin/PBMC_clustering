import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import NearestNeighbors


def log_transformation(df, features_to_ignore):
    """

    :param df: dataframe
        skup podataka
    :param features_to_ignore: list
        lista atributa koje ne treba transformisati
    :return: log transformed data
    """
    try:
        data_to_transform = df.drop(features_to_ignore, axis=1)
        transformed_data = data_to_transform.apply(lambda x: np.log2(x + 1))
        transformed_data[features_to_ignore] = df[features_to_ignore]
        return transformed_data

    except KeyError as e:
        print(e)


def drop_low_variance_columns(df, threshold=0.05):
    """

    :param df: dataframe
    :param threshold: float
    :return: dataframe
        novi dataframe bez kolona sa manjom varijansom od zadatog praga
    """
    selector = VarianceThreshold(threshold=threshold).fit(df)
    columns_to_keep = selector.get_support()
    df_without_low_var_columns = df.iloc[:, columns_to_keep]
    return df_without_low_var_columns


def get_numeric_columns(df):
    """

    :param df: dataframe
    :return: list
        vraca listu kolona ciji je tip numericki (float, int32, int...)
    """
    return df.select_dtypes(include=np.number).columns.tolist()


def drop_low_variance_columns_ignore_nonnumeric(df, threshold=0.05):
    """

    :param df: dataframe
    :param threshold: float
    :return: dataframe
        novi dataframe bez kolona sa manjom varijansom od zadatog praga
        ignorisuci ne-numericke kolone
    """

    numeric_columns = get_numeric_columns(df)
    numeric_data = df[numeric_columns]
    non_numeric_data = df.drop(columns=numeric_columns)

    selector = VarianceThreshold(threshold=threshold).fit(numeric_data)
    columns_to_keep = selector.get_support()
    df_without_low_var_columns = df.iloc[:, columns_to_keep]

    return df_without_low_var_columns.join(non_numeric_data)


def drop_zero_cols(df):
    """

    :param df: data frame
    :return: data frame without zero columns
    """
    non_zero_columns = (df.sum() != 0).values
    prepared_non_zero_cols = df.iloc[:, non_zero_columns]

    return prepared_non_zero_cols


def kth_nearest_neighbor(df, k):
    """

    :param df: dataframe
    :param k: int
    :return: list
        vraca listu koja sadrzi k-tog najblizeg suseda za svaku instancu iz dataframe-a
    """
    neighbors = NearestNeighbors(n_neighbors=k).fit(df)
    k_distance_matrix, _ = neighbors.kneighbors()
    kth_nearest_neighbors = list(map(lambda x: x[-1], k_distance_matrix))
    return kth_nearest_neighbors
