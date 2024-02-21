from components import EmbedNet
import torch
import torch.nn as nn
import torch.nn.functional as F

class DRP_MLP(nn.Module):
    def __init__(self,
                 drug_dim,
                 cell_dim,
                 drug_embed_dim=64,
                 cell_embed_dim=64,
                 embed_hid_dim=[512, 512],
                 fc_hid_dim=[512, 128],
                 dropout=0.5):
        super(DRP_MLP, self).__init__()
        
        self.drug_embed_net = EmbedNet(fc_in_dim=drug_dim, 
                                  fc_hid_dim=embed_hid_dim, 
                                  embed_dim=drug_embed_dim, 
                                  dropout=0.1)
        self.cell_embed_net = EmbedNet(fc_in_dim=cell_dim, 
                                  fc_hid_dim=embed_hid_dim, 
                                  embed_dim=cell_embed_dim, 
                                  dropout=0.1)
        
        # === Classifier === #
        self.fc = nn.Linear(drug_embed_dim+cell_embed_dim, fc_hid_dim[0])
        self.act = nn.ReLU()
        self.classifier = nn.ModuleList()
        
        for input_size, output_size in zip(fc_hid_dim, fc_hid_dim[1:]):
            self.classifier.append(
                nn.Sequential(nn.Linear(input_size, output_size),
                              nn.BatchNorm1d(output_size),
                              self.act,
                              nn.Dropout(p=dropout)
                             )
            )
        self.fc2 = nn.Linear(fc_hid_dim[-1], 1)
        for layer in self.classifier:
            nn.init.xavier_uniform_(layer[0].weight)
        
    def forward(self, data):
        drug_x, cell_x = data
        
        # === embed drug === #
        drug_x = self.drug_embed_net(drug_x)
        cell_x = self.cell_embed_net(cell_x)
        
        # === concat drug_x, cell_x === #
        input_vector = torch.cat((drug_x, cell_x), dim=1)
        x = F.relu(self.fc(input_vector))
        for fc in self.classifier:
            x = fc(x)
        output = self.fc2(x)
        return output
    