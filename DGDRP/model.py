from components import GNN, Cell_EmbedNet, EmbedNet
import torch.nn as nn
import torch.nn.functional as F
import torch

class HeteroNet(nn.Module):
    def __init__(self, 
                 gnn_embed_dim, 
                 cell_dim, 
                 drug_dim, 
                 cell_embed_dim=128, 
                 drug_embed_dim=128, 
                 predictor_hid_dims=[128, 128], 
                 predictor_dropout=0.6,
                 k=1000):
        super(HeteroNet, self).__init__()
        gnn_hid_dim = 512
        self.k = k

        self.gnn = GNN(feature_size=1, embedding_size=gnn_hid_dim, output_dim=gnn_embed_dim)
        self.cell_embed_net = Cell_EmbedNet(fc_in_dim=1, fc_hid_dim=[128, 256], embed_dim=gnn_embed_dim)

        self.cell_encoder = EmbedNet(fc_in_dim=cell_dim, fc_hid_dim=[128, 128], embed_dim=cell_embed_dim, dropout=0.2)
        self.drug_encoder = EmbedNet(fc_in_dim=drug_dim, fc_hid_dim=[128, 128], embed_dim=drug_embed_dim, dropout=0.2)

        # === Predictor === #
        self.fc = nn.Linear(cell_embed_dim+drug_embed_dim, predictor_hid_dims[0])    # + gnn_embed_dim
        self.act = nn.ReLU()
        self.classifier = nn.ModuleList()
        
        for input_size, output_size in zip(predictor_hid_dims, predictor_hid_dims[1:]):
            self.classifier.append(
                nn.Sequential(nn.Linear(input_size, output_size),
                              nn.BatchNorm1d(output_size),
                              self.act,
                              nn.Dropout(p=predictor_dropout)
                             )
            )
        self.fc2 = nn.Linear(predictor_hid_dims[-1], 1)
        for layer in self.classifier:
            nn.init.xavier_uniform_(layer[0].weight)


    def forward(self, data):
        graph, exp, drug = data

        # === Graph Embedding (Drug Target Information) === #
        graph_out = self.gnn(graph.x.unsqueeze(1), None, graph.edge_index, graph.batch)

        # === Gene Embeddings (Gene Expression Information) === #
        cell_embed_out = self.cell_embed_net(exp.unsqueeze(2))

        # ====== GNN + Cell EMB -> Top K Similarity Score Genes -> Cell Mask (Gene Selection based on Target Information) ====== #
        similarity_vector = torch.bmm(cell_embed_out, graph_out.unsqueeze(2))

        # ====== Top K Similarity score gene mask ====== #
        similarity_vector = F.softmax(similarity_vector.squeeze(2), dim=1)
        
        # === Get Top K values & indices === #
        _topk_vals, topk_indices = torch.topk(similarity_vector, self.k)

        # === Create mask : Top k values -> 1, else -> 0 === #
        mask = torch.zeros_like(similarity_vector)
        mask = mask.scatter_(1, topk_indices, 1.)

        masked_cell = exp * mask

        # ====== Cell ENC ====== #
        cell_embed = self.cell_encoder(masked_cell)
        drug_embed = self.drug_encoder(drug)

        # ====== Embedding Concat ====== #
        x = torch.cat([cell_embed, drug_embed], dim=1)
        
        # ====== Predictor ====== #
        x = F.relu(self.fc(x))
        for fc in self.classifier:
            x = fc(x)
        output = self.fc2(x)
        return output
    

class HeteroNet_EmbConcatTotal(nn.Module):
    def __init__(self, 
                 gnn_embed_dim, 
                 cell_dim, 
                 drug_dim, 
                 cell_embed_dim=128, 
                 drug_embed_dim=128, 
                 predictor_hid_dims=[128, 128], 
                 predictor_dropout=0.6,
                 k=1000):
        super(HeteroNet_EmbConcatTotal, self).__init__()
        gnn_hid_dim = 512
        self.k = k

        self.gnn = GNN(feature_size=1, embedding_size=gnn_hid_dim, output_dim=gnn_embed_dim)
        self.cell_embed_net = Cell_EmbedNet(fc_in_dim=1, fc_hid_dim=[128, 256], embed_dim=gnn_embed_dim)

        self.cell_encoder = EmbedNet(fc_in_dim=cell_dim, fc_hid_dim=[128, 128], embed_dim=cell_embed_dim, dropout=0.2)
        self.drug_encoder = EmbedNet(fc_in_dim=drug_dim, fc_hid_dim=[128, 128], embed_dim=drug_embed_dim, dropout=0.2)

        # === Predictor === #
        self.fc = nn.Linear(cell_embed_dim + drug_embed_dim + gnn_embed_dim, predictor_hid_dims[0])    # + gnn_embed_dim
        self.act = nn.ReLU()
        self.classifier = nn.ModuleList()
        
        for input_size, output_size in zip(predictor_hid_dims, predictor_hid_dims[1:]):
            self.classifier.append(
                nn.Sequential(nn.Linear(input_size, output_size),
                              nn.BatchNorm1d(output_size),
                              self.act,
                              nn.Dropout(p=predictor_dropout)
                             )
            )
        self.fc2 = nn.Linear(predictor_hid_dims[-1], 1)
        for layer in self.classifier:
            nn.init.xavier_uniform_(layer[0].weight)


    def forward(self, data):
        graph, exp, drug = data

        # === Graph Embedding (Drug Target Information) === #
        graph_out = self.gnn(graph.x.unsqueeze(1), None, graph.edge_index, graph.batch)

        # === Gene Embeddings (Gene Expression Information) === #
        cell_embed_out = self.cell_embed_net(exp.unsqueeze(2))

        # ====== GNN + Cell EMB -> Top K Similarity Score Genes -> Cell Mask (Gene Selection based on Target Information) ====== #
        similarity_vector = torch.bmm(cell_embed_out, graph_out.unsqueeze(2))

        # ====== Top K Similarity score gene mask ====== #
        similarity_vector = F.softmax(similarity_vector.squeeze(2), dim=1)
        
        # === Get Top K values & indices === #
        _topk_vals, topk_indices = torch.topk(similarity_vector, self.k)

        # === Create mask : Top k values -> 1, else -> 0 === #
        mask = torch.zeros_like(similarity_vector)
        mask = mask.scatter_(1, topk_indices, 1.)

        masked_cell = exp * mask

        # ====== Cell ENC ====== #
        cell_embed = self.cell_encoder(masked_cell)
        drug_embed = self.drug_encoder(drug)

        # ====== Embedding Concat ====== #
        x = torch.cat([cell_embed, drug_embed, graph_out], dim=1)
        
        # ====== Predictor ====== #
        x = F.relu(self.fc(x))
        for fc in self.classifier:
            x = fc(x)
        output = self.fc2(x)
        return output


    def get_mask(self, data):
        graph, exp, drug = data

        # === Graph Embedding (Drug Target Information) === #
        graph_out = self.gnn(graph.x.unsqueeze(1), None, graph.edge_index, graph.batch)

        # === Gene Embeddings (Gene Expression Information) === #
        cell_embed_out = self.cell_embed_net(exp.unsqueeze(2))

        # ====== GNN + Cell EMB -> Top K Similarity Score Genes -> Cell Mask (Gene Selection based on Target Information) ====== #
        similarity_vector = torch.bmm(cell_embed_out, graph_out.unsqueeze(2))

        # ====== Top K Similarity score gene mask ====== #
        similarity_vector = F.softmax(similarity_vector.squeeze(2), dim=1)
        
        # === Get Top K values & indices === #
        _topk_vals, topk_indices = torch.topk(similarity_vector, self.k)

        # === Create mask : Top k values -> 1, else -> 0 === #
        mask = torch.zeros_like(similarity_vector)
        mask = mask.scatter_(1, topk_indices, 1.)

        masked_cell = exp * mask

        # ====== Cell ENC ====== #
        cell_embed = self.cell_encoder(masked_cell)
        drug_embed = self.drug_encoder(drug)

        # ====== Embedding Concat ====== #
        x = torch.cat([cell_embed, drug_embed, graph_out], dim=1)
        
        # ====== Predictor ====== #
        x = F.relu(self.fc(x))
        for fc in self.classifier:
            x = fc(x)
        output = self.fc2(x)
        return mask # output, similarity_vector, mask, topk_indices