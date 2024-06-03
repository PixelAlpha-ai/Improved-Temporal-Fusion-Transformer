import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from models import Informer
import os

# Define global parameters
x_stand = StandardScaler()
y_stand = StandardScaler()
s_len = 64
pre_len = 5
batch_size = 32
device = "cuda" if torch.cuda.is_available() else "cpu"
path_testing_data = 'datas_test/'
model_save_path = 'trained_models/'  # Directory to save the model

def create_data(datas):
    values = []
    labels = []
    lens = datas.shape[0]
    datas = datas.values
    for index in range(0, lens - pre_len - s_len):
        value = datas[index:index + s_len, [0, 1, 2, 3, 4]]
        # Create dummy labels to match the original data structure
        label = datas[index + s_len - pre_len:index + s_len + pre_len, [0, 4]]
        values.append(value)
        labels.append(label)
    return values, labels

def read_data(name_data_file, last_known_date):
    datas = pd.read_csv(f"{path_testing_data}\\{name_data_file}.csv")

    # Filter the data based on the provided date
    datas['Date'] = pd.to_datetime(datas['Date'])
    datas = datas[datas['Date'] <= last_known_date]

    # Revert the Date column back to string format if needed
    datas['Date'] = datas['Date'].dt.strftime('%Y-%m-%d')

    # Process data
    datas = datas[["Date", "Open", "High", "Low", "Close", "Volume"]]
    datas.fillna(0, inplace=True)
    xs = datas.values[:, [1, 2, 3, 4]]
    x_stand.fit(xs)
    values, labels = create_data(datas)

    return values, labels

class AmaData(Dataset):
    def __init__(self, values, labels):
        self.values, self.labels = values, labels

    def __len__(self):
        return len(self.values)

    def create_time(self, data):
        time = data[:, 0]
        time = pd.to_datetime(time)
        week = np.int32(time.dayofweek)[:, None]
        month = np.int32(time.month)[:, None]
        day = np.int32(time.day)[:, None]
        time_data = np.concatenate([month, week, day], axis=-1)
        return time_data

    def __getitem__(self, item):
        value = self.values[item]
        label = self.labels[item]
        value_t = self.create_time(value)
        label_t = self.create_time(label)
        value = x_stand.transform(value[:, 1:])
        label = y_stand.transform(label[:, 1][:, None])
        value = np.float32(value)
        label = np.float32(label)
        return value, label, value_t, label_t

def infer_model(name_data_file, test_x, test_y):
    test_data = AmaData(test_x, test_y)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    model = Informer()
    model.load_state_dict(torch.load(model_save_path + f'model_{name_data_file}.pth'))
    model.to(device)

    model.eval()
    predictions = []

    with torch.no_grad():
        for x, y, xt, yt in test_loader:
            mask = torch.zeros_like(y)[:, pre_len:].to(device)
            x, y, xt, yt = x.to(device), y.to(device), xt.to(device), yt.to(device)
            dec_y = torch.cat([y[:, :pre_len], mask], dim=1)
            logits = model(x, xt, dec_y, yt)
            predictions.extend(logits.cpu().numpy())

    return predictions

if __name__ == '__main__':
    # Example usage: predict prices for 'AAPL' up to '2024-05-20'
    name_symbol = 'AAPL'
    last_known_date = pd.Timestamp('2024-05-20')

    # Read the data
    test_x, test_y = read_data(name_symbol, last_known_date)

    # Run inference with the pre-trained model
    if len(test_x) > 0:
        predictions = infer_model(name_symbol, test_x, test_y)
        print("Predictions processed.")
        print(predictions)
    else:
        print("Not enough data to form a test set with the specified 'last_known_date'.")
