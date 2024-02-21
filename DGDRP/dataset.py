import os
from torch_geometric.data import Dataset, Data
import pandas as pd
import torch
# from tqdm import tqdm
# from scipy import sparse
# import numpy as np


class DrugResponse_Dataset(Dataset):
    def __init__(self, response_data, expression_data, drug_data, adjacency_matrix, root='/data/project/minwoo/', data_type='50_indirect_targets', transform=None, pre_transform=None): # , response_type='IC50'
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.root = root

        # self.response_type = response_type
        self.data_type = data_type

        self.response_data = response_data
        self.expression_data = expression_data.float()
        self.drug_data = drug_data.float()
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
        return os.path.join(self.root, 'Data', 'drug_response', 'cv_drug', 'train_valid_response_by_drug_cv00.tsv')

    @property
    def processed_file_names(self):
        """
        If these files are found in raw_dir, processing is skipped.
        """
        return [os.path.join(self.processed_dir, f'graph_{self.data_type}_{idx}.pt') for idx, _ in self.response_data.iterrows()]

    def download(self):
        pass

    def process(self):
        assert len(self.response_data) == len(self.expression_data)
        raise RuntimeError('Dataset should be already processed.')
        # for index, row in tqdm(self.response_data.iterrows()):
        #     exp = self.expression_data.iloc[index]
        #     drug = row['drug_name']
        #     responses = row[self.response_type]

        #     # ====== Get Node Features ====== #
        #     node_feats = self._get_node_features(exp)

        #     # ====== Get Edges Connections ====== #
        #     edge_indices = self._get_edge_index(drug)

        #     # ====== Get Labels ====== #
        #     label = torch.Tensor([responses]).to(torch.float)

        #     # ====== Create Data Object ====== #
        #     data = Data(x=node_feats, edge_index=edge_indices, y=label)

        #     torch.save(data, os.path.join(self.processed_dir, f'graph_IC50_{self.data_type}_{index}.pt'))


    # def _get_node_features(self, exp):
    #     node_feats = []
    #     for node in self.node_list:
    #         if node in list(exp.index):
    #             node_exp = exp[node]
    #             node_feats.append(node_exp)
    #         else:
    #             node_feats.append(0)
    #     node_feats = torch.Tensor(node_feats).to(torch.float)
    #     return node_feats


    # def _get_edge_index(self, drug):
    #     adj_fpath = os.path.join(self.datadir, 'feature_selection', 'drug_networks', f'{drug}_adjacency_matrix.tsv')
    #     adj_matrix = pd.read_csv(adj_fpath, sep = '\t', index_col = 0, header=0)
    #     sparse_mx = sparse.csr_matrix(adj_matrix).tocoo()
    #     sparse_mx = sparse_mx.astype(np.float32)
    #     edge_indices = np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    #     edge_indices = torch.from_numpy(edge_indices)
    #     return edge_indices


    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        graph_data = torch.load(os.path.join(self.processed_dir, f'graph_{self.data_type}_{idx}.pt'))
        exp = self.expression_data[idx]
        drug = self.drug_data[idx]
        return (graph_data, exp, drug), graph_data.y


    def len(self):
        return self.response_data.shape[0]
