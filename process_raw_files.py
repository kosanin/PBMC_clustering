from src.GeneMetadataHandler import GeneMetadataHandler
from src.preprocessing import (drop_non_common_genes_and_get_genes_ids
            , concatenate_ensg_and_gene_columns
            , rename_cells
            , convert_columns_to_unit16)
import pandas as pd
import os
import sys
import json

common_human_list_path = ""
pbmc_metadata_path = ""

if len(sys.argv) >= 4:
    common_human_list_path = sys.argv[1]
    pbmc_metadata_path = sys.argv[2]
else:
    print("Use ./process_raw_files.py common_human_path pbmc_metadata_path dir_path - containing raw data ")
    sys.exit(1)


gene_intersection = set()
gene_data_handler = GeneMetadataHandler(common_human_list_path, pbmc_metadata_path)
log_data = {}

def process_file(filepath):

    global gene_intersection, log_data

    print("Reading file...")
    df = pd.read_csv(filepath)
    df = convert_columns_to_unit16(df)

    sample_id = gene_data_handler.get_sample_id(filepath)
    genome = gene_data_handler.get_genome(sample_id)

    file_shape_info = {}
    file_shape_info['raw_shape'] = df.shape

    df = drop_non_common_genes_and_get_genes_ids(df, gene_data_handler.get_gene_lookup_columns(genome))
    df = concatenate_ensg_and_gene_columns(df, gene_data_handler.get_gene_lookup_column_name(genome))
    df.columns = rename_cells(df.columns, sample_id)

    file_shape_info['no_uncommon_genes'] = df.shape

    df = df.set_index(df.columns[0])
    df = df.transpose()

    file_shape_info['transposed_shape'] = df.shape
    log_data[sample_id] = file_shape_info

    if len(gene_intersection):
        gene_intersection &= set(df.columns.values)
    else:
        # fill empty set with genes from first file read
        gene_intersection |= set(df.columns.values)

    print("Writing file...")
    df.to_csv("ir_data/" + sample_id + ".csv", index_label='cell_id')


top_dir = sys.argv[3]
for root, _, files in os.walk(top=top_dir):
    for file in files:
        print('################################################################')
        print("Processing " + file + " ...")
        process_file(os.path.join(root, file))


with open("log_data.json", "w") as f:
    print("Dumping log data to json file...")
    json.dump(log_data, f)

with open("gene_intersection.json", "w") as f:
    print("Dumping gene intersection to json file")
    json.dump(list(gene_intersection), f)
