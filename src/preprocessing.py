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


def highly_correlated_cols(df):
    """

    :param df: dataframe
    :return: list
        vraca listu kolona za koje postoje neke druge sa kojima su jako korelisane,
        ukradena fja od Veljkovica
    """
    critical_columns = []
    correlation_matrix = np.corrcoef(df, rowvar=False)

    number_of_columns = correlation_matrix.shape[0]
    for ith_column in range(number_of_columns):
        for jth_column in range(ith_column + 1, number_of_columns):
            if np.allclose(abs(correlation_matrix[ith_column][jth_column]), 1):
                critical_columns.append(df.columns[ith_column])

    return critical_columns


def convert_columns_to_unit16(df):
    """
    :df: DataFrame
    :return: DataFrame
    Converts numeric columns to unit16 type; Ignores first column(assuming its id column)
    """
    print("Converting numeric types to uint16...")
    id_column = df.columns[0]
    id_column_data = df[id_column]

    # ova linija menja prosledjeni df
    df.drop(columns=[id_column], inplace=True)
    df = df.astype('uint16')
    df.insert(0, column=id_column, value=id_column_data)
    return df


def drop_non_common_genes_and_get_genes_ids(df, reference_data):
    """
    :df: DataFrame
        expected to have 'Index' or gene names column
    :reference_data: DataFrame
        output from GeneMetadataHandler.get_gene_lookup_columns method
    :return: DataFrame
        DataFrame without untracked genes(genes that are not in common_human_list) 
        and with ENSG_ID or gene column added, depending on df
    """
    print("Dropping uncommon genes...")
    gene_lookup_column_name = reference_data.columns[1]
    join_column = 'ENSG_ID' if df.columns[0] == 'Index' else gene_lookup_column_name
    
    # performing inner join to remove untracked genes(genes that are not in common_human_list) from df
    return reference_data.join(other=df.set_index(df.columns[0])
                              ,on=join_column
                              ,how='inner')


def concatenate_ensg_and_gene_columns(df, gene_lookup_column):
    """
    :df: DataFrame
    :return: DataFrame
        Constructs gene_id column by concatenating ENSG_ID and gene columns
    """
    print("Creating key column gene_id")
    df.insert(0, 'gene_id', df['ENSG_ID'] + "_" + df[gene_lookup_column])
    df.drop(columns=['ENSG_ID', gene_lookup_column], inplace=True)
    return df


def rename_cells(columns, sample_id):
    print("Renaming cells...")
    number_of_cells = columns.shape[0] - 1  # -1 za Index(gene) kolonu
    return [columns[0]] + [sample_id + "_" + str(cell_number) for cell_number in range(1, number_of_cells + 1)]
    
    
    
def filter_by_percentage(df, p=1):
    """
    :param df: DataFrame
    :param p: Positive Int
    :return: DataFrame
    returns DataFrame without columns that have less than p% non-zero values
    """
    columns_non_zero_percentages = (df != 0) \
                                        .sum() \
                                        .apply(lambda x: x / df.shape[1] * 100)
    columns_to_keep = (columns_non_zero_percentages > p).values
    return df.iloc[:, columns_to_keep]
    
    
def filter_cells(df, sum_of_gene_exprs_lower_limit=1000, number_of_expressed_genes_lower_limit=500):
    """
    :param df: DataFrame
    :sum_of_gene_exprs_lower_limit: Positive Integer
    :number_of_expressed_genes_lower_limit: Positive Integer
    """
    number_of_expressed_genes_per_cell = (df != 0).sum(axis=1)
    sum_of_gene_exprssions_per_cell = df.sum(axis=1)
    cells_to_keep = ((sum_of_gene_exprssions_per_cell > sum_of_gene_exprs_lower_limit) & 
                     (number_of_expressed_genes_per_cell > number_of_expressed_genes_lower_limit)).values
    
    return df.iloc[cells_to_keep, :]

