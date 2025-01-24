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
        return (self.data_length // 24) * self.num_cities

    def __getitem__(self, idx):
        city_id = idx // (self.data_length * 24)
        temp_id = idx % (self.data_length // 24) * 24
        data_id = (temp_id) + (3*24)

        data_triplet = []
        for i in range(1, 73):
            city = self.ids[city_id]
            coords = self.coordinates['Latitude'][city_id], self.coordinates['Longitude'][city_id]
            datetime = self.datetimes.iloc[data_id - i, 1:].values
            
            data = []
            for df in self.data:
                data.append(df[city][data_id - i])
            data_triplet.append((*coords, *datetime, *data))

        data_output = []
        for i in range(24, 48, 1):
            city = self.ids[city_id]
            coords = self.coordinates['Latitude'][city_id], self.coordinates['Longitude'][city_id]
            datetime = self.datetimes.iloc[data_id + i, 1:].values

            data = []
            for df in self.data:
                data.append(df[city][data_id + i])
            data_output.append((*coords, *datetime, *data))

        avg_temperature = sum(row[13] for row in data_output) / len(data_output) # max_temp = 321.22 min_temp = 242.337
        max_wind_speed = max(row[-1] for row in data_output)
        denormalized_wind_speed = max_wind_speed*50 # max_wind = 50.0 min_speed = 0.0

        if denormalized_wind_speed >= 6:
            classification = [0, 1]
        else:
            classification = [1, 0]

        return torch.Tensor(data_triplet), torch.Tensor((avg_temperature, *classification)).unsqueeze(0)# X, Y