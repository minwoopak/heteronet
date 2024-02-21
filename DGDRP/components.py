import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TopKPooling # TransformerConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class GNN(nn.Module):
    def __init__(self, feature_size, embedding_size = 512, output_dim = 256):
        super(GNN, self).__init__()

        # GNN layers
        self.conv1 = GATConv(feature_size, embedding_size, heads=3, dropout=0.6)
        self.head_transform1 = nn.Linear(embedding_size*3, embedding_size)
        self.bn1 = nn.BatchNorm1d(embedding_size)
        self.pool1 = TopKPooling(embedding_size, ratio=0.8)

        self.conv2 = GATConv(embedding_size, embedding_size, heads=3, dropout=0.6)
        self.head_transform2 = nn.Linear(embedding_size*3, embedding_size)
        self.bn2 = nn.BatchNorm1d(embedding_size)
        self.pool2 = TopKPooling(embedding_size, ratio=0.5)
        
        self.conv3 = GATConv(embedding_size, embedding_size, heads=3, dropout=0.6)
        self.head_transform3 = nn.Linear(embedding_size*3, embedding_size)
        self.bn3 = nn.BatchNorm1d(embedding_size)
        self.pool3 = TopKPooling(embedding_size, ratio=0.2)

        # Linear Layers
        self.linear1 = nn.Linear(embedding_size*2, 512)
        self.linear2 = nn.Linear(512, output_dim)

    def forward(self, x, _edge_attr, edge_index, batch_index):
        # First Block
        x = self.conv1(x, edge_index)
        x = F.relu(self.head_transform1(x))
        x, edge_index, _edge_attr, batch_index, _, _ = self.pool1(x, edge_index, None, batch_index)
        x1 = torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)

        # Second Block
        x = self.conv2(x, edge_index)
        x = F.relu(self.head_transform2(x))
        x, edge_index, _edge_attr, batch_index, _, _ = self.pool2(x, edge_index, None, batch_index)
        x2 = torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)

        # Third Block
        x = self.conv3(x, edge_index)
        x = F.relu(self.head_transform3(x))
        x, edge_index, _edge_attr, batch_index, _, _ = self.pool3(x, edge_index, None, batch_index)
        x3 = torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)

        # Concat pooled vectors
        x = x1 + x2 + x3

        # Output block
        x = self.linear1(x).relu()
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.linear2(x)

        return x


# ====== Cell EMB ====== #
class Cell_EmbedNet(nn.Module):
    def __init__(self, fc_in_dim=1024, fc_hid_dim=[512, 512], embed_dim=512, dropout=0.5):
        super(Cell_EmbedNet, self).__init__()
        self.fc_hid_dim = fc_hid_dim
        self.fc = nn.Linear(fc_in_dim, self.fc_hid_dim[0])
        self.act = nn.ReLU()
        self.dropout = dropout
        self.classifier = nn.ModuleList()
        
        for input_size, output_size in zip(self.fc_hid_dim, self.fc_hid_dim[1:]):
            self.classifier.append(
                nn.Sequential(nn.Linear(input_size, output_size),
                              self.act,
                              nn.Dropout(p=self.dropout)
                             )
            )
        self.fc2 = nn.Linear(self.fc_hid_dim[-1], embed_dim)
        for layer in self.classifier:
            nn.init.xavier_uniform_(layer[0].weight)
    
    def forward(self, x):
        x = F.relu(self.fc(x))
        for fc in self.classifier:
            x = fc(x)
        x = self.fc2(x)
        return x


class EmbedNet(nn.Module):
    def __init__(self, fc_in_dim=1024, fc_hid_dim=[512, 512], embed_dim=512, dropout=0.5):
        super(EmbedNet, self).__init__()
        self.fc_hid_dim = fc_hid_dim
        self.fc = nn.Linear(fc_in_dim, self.fc_hid_dim[0])
        self.act = nn.ReLU()
        self.dropout = dropout
        self.classifier = nn.ModuleList()
        
        for input_size, output_size in zip(self.fc_hid_dim, self.fc_hid_dim[1:]):
            self.classifier.append(
                nn.Sequential(nn.Linear(input_size, output_size),
                              nn.BatchNorm1d(output_size),
                              self.act,
                              nn.Dropout(p=self.dropout)
                             )
            )
        self.fc2 = nn.Linear(self.fc_hid_dim[-1], embed_dim)
        for layer in self.classifier:
            nn.init.xavier_uniform_(layer[0].weight)
    
    def forward(self, x):
        x = F.relu(self.fc(x))
        for fc in self.classifier:
            x = fc(x)
        x = self.fc2(x)
        return x


# class HardArgmax(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input: torch.Tensor, dim=-1):
#         index = torch.argmax(input.detach(), dim=dim)
#         ctx.save_for_backward(index)
#         ctx.dim = dim
#         one_hot = torch.nn.functional.one_hot(index.flatten(), num_classes=input.shape[dim]).float()
#         return one_hot.view(*input.shape)

#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output.clone(), None

# def hard_argmax(input: torch.Tensor,dim=-1):
#     return HardArgmax.apply(input,dim)

