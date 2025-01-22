import torch
from torch.utils.data import Dataset

from data_processing import load_data

class WeatherDataset(Dataset):
    def __init__(self, data_dir='./data', normalize_data=True):
        self.ids, self.coordinates, self.datetimes, *self.data = load_data(data_dir)
        
        print(self.datetimes)

        self.data_length = len(self.datetimes) - (4*24)
        self.num_cities = len(self.ids)        

        self.device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return self.data_length * self.num_cities

    def __getitem__(self, idx):
        city_id = idx // self.data_length
        temp_id = idx % self.data_length
        to_full_day = 24 - (temp_id % 24)
        data_id = (temp_id + to_full_day) + (3*24)

        data_triplet = []
        for i in range(1, 73):
            city = self.ids[city_id]
            coords = self.coordinates['Latitude'][city_id], self.coordinates['Longitude'][city_id]
            datetime = self.datetimes.iloc[data_id - i, 1:].values
            
            data = []
            for df in self.data:
                data.append(df[city][data_id - i])
            data_triplet.append((*coords, *datetime, *data))

        city = self.ids[city_id]
        coords = self.coordinates['Latitude'][city_id], self.coordinates['Longitude'][city_id]
        datetime = self.datetimes.iloc[data_id + 1, 1:].values
            
        data = []
        for df in self.data:
            data.append(df[city][data_id + 1])
        
        return torch.Tensor(data_triplet), torch.Tensor((*coords, *datetime, *data)).unsqueeze(0)# X, Y