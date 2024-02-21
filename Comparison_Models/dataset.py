from torch.utils.data import Dataset
import torch

class DRPDataset(Dataset):
    def __init__(self, response_data, expression_data, drug_data, response_type='IC50'):
        self.response_type = response_type

        self.response_data = response_data
        self.expression_data = expression_data.float()
        self.drug_data = drug_data.float()

    def __len__(self):
        return len(self.response_data)

    def __getitem__(self, idx):
        response = torch.Tensor([self.response_data.iloc[idx][self.response_type]]).to(torch.float)
        cell_x = self.expression_data[idx]
        drug_x = self.drug_data[idx]

        return (drug_x, cell_x), response
