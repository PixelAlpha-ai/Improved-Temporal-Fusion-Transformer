import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from models import Informer
import matplotlib.pyplot as plt

# Define global parameters
train_size = 0.7  # Proportion of data to use for training
test_size = 0.15  # Proportion of data to use for the OOS test set
val_size = 0.15  # Proportion of data to use for validation
x_stand = StandardScaler()
y_stand = StandardScaler()
s_len = 64
pre_len = 5
batch_size = 32
device = "cuda"
lr = 5e-5
epochs = 100
model_save_path = 'trained_models/'  # Directory to save the model


def create_data(datas):
    values = []
    labels = []
    lens = datas.shape[0]
    datas = datas.values
    for index in range(0, lens - pre_len - s_len):
        value = datas[index:index + s_len, [0, 2, 3, 4, 5]]
        label = datas[index + s_len - pre_len:index + s_len + pre_len, [0, 1]]
        values.append(value)
        labels.append(label)
    return values, labels


def read_data(name_data_file):
    datas = pd.read_csv(f"datas\\{name_data_file}.csv")
    datas.pop("Adj Close")
    datas.fillna(0)
    xs = datas.values[:, [2, 3, 4, 5]]
    ys = datas.values[:, 1]
    x_stand.fit(xs)
    y_stand.fit(ys[:, None])
    values, labels = create_data(datas)

    # Split off the OOS test set first
    train_val_x, oos_x, train_val_y, oos_y = train_test_split(values, labels, test_size=test_size)

    # Split the remaining data into train and validation sets
    train_x, val_x, train_y, val_y = train_test_split(train_val_x, train_val_y, train_size=train_size/(train_size + val_size))

    return train_x, val_x, train_y, val_y, oos_x, oos_y

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


def train_model(name_data_file, train_x, train_y, val_x, val_y):
    # train_x, val_x, train_y, val_y, oos_x, oos_y = read_data(name_data_file)
    train_data = AmaData(train_x, train_y)
    train_data = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_data = AmaData(val_x, val_y)
    val_data = DataLoader(val_data, shuffle=True, batch_size=batch_size)
    # oos_data = AmaData(oos_x, oos_y)
    # oos_data = DataLoader(oos_data, shuffle=False, batch_size=batch_size)

    model = Informer()
    model.to(device)
    loss_fc = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_data, desc=f"Training Epoch {epoch + 1}")
        for step, (x, y, xt, yt) in enumerate(pbar):
            mask = torch.zeros_like(y)[:, pre_len:].to(device)
            x, y, xt, yt = x.to(device), y.to(device), xt.to(device), yt.to(device)
            dec_y = torch.cat([y[:, :pre_len], mask], dim=1)
            logits = model(x, xt, dec_y, yt)
            loss = loss_fc(logits, y[:, pre_len:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            pbar = tqdm(val_data, desc=f"Validation Epoch {epoch + 1}")
            for step, (x, y, xt, yt) in enumerate(pbar):
                mask = torch.zeros_like(y)[:, pre_len:].to(device)
                x, y, xt, yt = x.to(device), y.to(device), xt.to(device), yt.to(device)
                dec_y = torch.cat([y[:, :pre_len], mask], dim=1)
                logits = model(x, xt, dec_y, yt)
                loss = loss_fc(logits, y[:, pre_len:])
                val_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
        val_loss /= len(val_data)
        print(f"Validation Loss after Epoch {epoch + 1}: {val_loss}")

    print("Training Complete. Saving the model...")

    save_path = model_save_path + f'model_{name_data_file}.pth'

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def infer_model(name_data_file, oos_x, oos_y):
    oos_data = AmaData(oos_x, oos_y)
    oos_data = DataLoader(oos_data, shuffle=False, batch_size=batch_size)

    model = Informer()
    model.load_state_dict(torch.load(model_save_path + f'model_{name_data_file}.pth'))
    model.to(device)

    loss_fc = nn.MSELoss()
    model.eval()
    oos_loss = 0.0
    actuals, predictions = [], []
    with torch.no_grad():
        for x, y, xt, yt in oos_data:
            mask = torch.zeros_like(y)[:, pre_len:].to(device)
            x, y, xt, yt = x.to(device), y.to(device), xt.to(device), yt.to(device)
            dec_y = torch.cat([y[:, :pre_len], mask], dim=1)
            logits = model(x, xt, dec_y, yt)
            loss = loss_fc(logits, y[:, pre_len:])
            oos_loss += loss.item()
            actuals.append(y.cpu().numpy())
            predictions.append(logits.cpu().numpy())
    oos_loss /= len(oos_data)
    print(f"OOS Test Loss: {oos_loss}")

    # Convert lists to arrays for easier handling
    actuals = np.concatenate(actuals, axis=0)
    predictions = np.concatenate(predictions, axis=0)

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(actuals[:, 0], label='Actual')
    plt.plot(predictions[:, 0], label='Predicted')
    plt.title('Out-of-Sample Predictions vs Actuals')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    name_data_file = 'Amazon'

    # Read the data
    train_x, val_x, train_y, val_y, oos_x, oos_y = read_data(name_data_file)

    # Uncomment the next line to train the model
    # train_model(name_data_file, train_x, train_y, val_x, val_y)

    # Run inference with the pre-trained model
    infer_model(name_data_file, oos_x, oos_y)