import pandas as pd
import re

class GeneMetadataHandler:
    def __init__(self, common_human_path, pbmc_metadata_path):
        self._common_human = pd.read_csv(common_human_path)
        self._pbmc_metadata = pd.read_csv(pbmc_metadata_path)
        self._sample_id_regex = re.compile("GSM[0-9]{7}")
    
    def get_sample_id(self, filename):
        match = self._sample_id_regex.search(filename)
        if match:
            return match.group()
        else:
            #TODO handlovati ovaj exception
            return ""
    
    def get_genome(self, sample_id):
        # TODO pretpostavlja da ce sve biti OK
        sample_metadata = self._pbmc_metadata.loc[self._pbmc_metadata['SAMPLE'] == sample_id]
        sample_genome = sample_metadata['GENOME'].values[0]
        return sample_genome
    
    def get_gene_lookup_column_name(self, sample_genome):
        if sample_genome not in self._common_human.columns:
            return 'Ensembl_GRCh38.p12_rel94'
        else:
            return sample_genome
        
    def get_gene_lookup_columns(self, sample_genome):
        lookup_column = self.get_gene_lookup_column_name(sample_genome)
        return self._common_human[['ENSG_ID', lookup_column]]
    