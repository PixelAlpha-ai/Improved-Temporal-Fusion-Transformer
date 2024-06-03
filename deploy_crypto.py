import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
import os
from models import Informer

# Define global parameters
x_stand = StandardScaler()
s_len = 64  # Sequence length
device = "cuda" if torch.cuda.is_available() else "cpu"

model_save_path = 'trained_models'
path_test_data = 'datas_test'

def read_data(filepath, last_known_date):
    datas = pd.read_csv(filepath)
    datas['Date'] = pd.to_datetime(datas['Date'])
    datas = datas[datas['Date'] <= last_known_date]
    datas.fillna(0, inplace=True)

    # Assuming datas includes a 'Date' and financial metrics columns
    xs = datas[['Open', 'High', 'Low', 'Close', 'Volume']].values
    x_stand.fit(xs)

    # Create test data sequences
    test_x = create_data(datas)
    return test_x


def create_data(datas):
    values = []
    lens = len(datas)
    if lens > s_len:
        for index in range(0, lens - s_len + 1):
            value = datas.iloc[index:index + s_len][['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].values
            values.append(value)
    return np.array(values)


class AmaData(Dataset):
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        value = self.values[item]
        # Extract and transform the date to be usable as an input
        date_values = pd.to_datetime(value[:, 0])
        time_features = np.stack((date_values.month, date_values.day, date_values.weekday()), axis=1)
        numeric_features = x_stand.transform(value[:, 1:].astype(float))

        combined_features = np.hstack((time_features, numeric_features))
        combined_features = torch.tensor(combined_features, dtype=torch.float32)

        return combined_features


def infer_model(model, test_x):
    test_data = AmaData(test_x)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    model.eval()
    predictions = []
    with torch.no_grad():
        for x in test_loader:
            x = x.to(device)
            outputs = model(x)
            predictions.extend(outputs.cpu().numpy())
    return predictions


# Example of using the functions
if __name__ == '__main__':

    # input to infer_model
    name_symbol = 'AAPL'
    last_known_date = pd.Timestamp('2024-05-20')


    # Assuming the Informer model is defined somewhere and loaded here
    model = Informer()
    model.load_state_dict(torch.load((model_save_path + f'\\model_{name_symbol}.pth')))
    model.to(device)

    # read the input data
    filepath = os.path.join(path_test_data, f'{name_symbol}.csv')
    test_x = read_data(filepath, last_known_date)

    if len(test_x) > 0:
        predictions = infer_model(model, test_x)
        print("Predictions processed.")
    else:
        print("Not enough data to form a test set with the specified 'last_known_date'.")
