import os
import random
import argparse
parser = argparse.ArgumentParser()
args, _ = parser.parse_known_args()
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
#from collections import defaultdict
#import gseapy as gp
from tqdm import tqdm
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data

from tqdm import tqdm
from scipy import sparse

#from heteronet.dataset import DrugResponse_Dataset

# ====== Random Seed Initialization ====== #
def seed_everything(seed = 3078):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything()


parser.add_argument('--datadir', type=str, default='/data/project/inyoung/DGDRP/Data')
parser.add_argument('--n_indirect_targets', type=int, default=20)
parser.add_argument('--data_type', type=str, default='20_indirect_targets')

args = parser.parse_args()

response_fpath = os.path.join(args.datadir, 'response_data_total.tsv')
expression_fpath = os.path.join(args.datadir, 'expression_10k_genes_data_total.tsv')
drug_fp_fpath = os.path.join(args.datadir, 'drug_data_total.tsv')

response_data = pd.read_csv(response_fpath, sep='\t')
response_data = response_data.reset_index().rename(columns={'index':'idx'})
expression_data = pd.read_csv(expression_fpath, sep='\t', index_col=0, header=0)
drug_data = pd.read_csv(drug_fp_fpath, sep='\t', index_col=0)

root = os.path.join(args.datadir, 'graph_pyg', args.data_type)


# =========================== #
# ====== Dataset Class ====== #
# =========================== #
class DrugResponse_Dataset(Dataset):
    def __init__(self, response_data, expression_data, drug_data, adjacency_matrix, 
                 root='/data/project/inyoung/', response_type='IC50', 
                 data_type='indirect_top_20', transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.root = root
        self.response_type = response_type
        self.data_type = data_type

        self.response_data = response_data
        self.expression_data = expression_data
        self.drug_data = drug_data

        self.adjacency_matrix = adjacency_matrix

        self.node_list = self.adjacency_matrix.columns.to_list()

        super(DrugResponse_Dataset, self).__init__(root, transform, pre_transform)
        print(self.raw_dir)
        print(self.processed_dir)
        

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return os.path.join(self.root, 'Data', 'response_data_total.tsv')

    @property
    def processed_file_names(self):
        """
        If these files are found in raw_dir, processing is skipped.
        """
        return [os.path.join(self.processed_dir, f'graph_{self.data_type}_{index}.pt') for index, _ in self.response_data.iterrows()]

    def download(self):
        pass

    def process(self):
        assert len(self.response_data) == len(self.expression_data)

        for index, row in tqdm(self.response_data.iterrows()):
            exp = self.expression_data.iloc[index]
            drug = row['drug_name']
            responses = row[self.response_type]

            # ====== Get Node Features ====== #
            node_feats = self._get_node_features(exp)

            # ====== Get Edges Connections ====== #
            edge_indices = self._get_edge_index(drug)

            # ====== Get Labels ====== #
            label = torch.Tensor([responses]).to(torch.float)

            # ====== Create Data Object ====== #
            data = Data(x=node_feats, edge_index=edge_indices, y=label)

            torch.save(data, os.path.join(self.processed_dir, f'graph_{self.data_type}_{index}.pt'))


    def _get_node_features(self, exp):
        node_feats = []
        for node in self.node_list:
            if node in list(exp.index):
                node_exp = exp[node]
                node_feats.append(node_exp)
            else:
                node_feats.append(0)
        node_feats = torch.Tensor(node_feats).to(torch.float)
        return node_feats


    def _get_edge_index(self, drug):
        adj_fpath = os.path.join(args.datadir, 
                                f'drug_networks_{self.data_type}',                  ####################################
                                f'{drug}_adjacency_matrix_{self.data_type}.tsv')    ####################################
        adj_matrix = pd.read_csv(adj_fpath, sep = '\t', index_col = 0, header=0)
        sparse_mx = sparse.csr_matrix(adj_matrix).tocoo()
        sparse_mx = sparse_mx.astype(np.float32)
        edge_indices = np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        edge_indices = torch.from_numpy(edge_indices)
        return edge_indices


    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        # graph_data = torch.load(os.path.join(self.processed_dir, f'graph_{self.data_type}_{index}.pt'))
        # exp = torch.tensor(self.expression_data.iloc[idx], dtype=torch.float)
        # drug = torch.tensor(self.drug_data.iloc[idx], dtype=torch.float)
        # return (graph_data, exp, drug), graph_data.y
        pass


    def len(self):
        return self.response_data.shape[0]
    

# =============================== #
# ====== Construct Dataset ====== #
# =============================== #
template_adj_fpath = os.path.join(args.datadir, f'template_adjacency_matrix_{args.data_type}.tsv')
adjacency_matrix = pd.read_csv(template_adj_fpath, sep='\t', index_col=0, header=0)

total_dataset = DrugResponse_Dataset(response_data, expression_data, drug_data, 
                                     adjacency_matrix, root=root, data_type=args.data_type)