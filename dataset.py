import torch
from torch.utils.data import Dataset

from data_processing import load_data

class WeatherDataset(Dataset):
    def __init__(self, data_dir='./data', transform=None):
        self.data = load_data(data_dir)
        
        self.transform = transform
        self.device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pass
