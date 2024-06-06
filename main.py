"""
Main function for daily run
"""
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import os
import joblib
import yfinance as yf
from models import Informer
from config import symbols_daily_run
from datetime import datetime, timedelta


x_stand = StandardScaler()
y_stand = StandardScaler()
s_len = 64
batch_size = 64
pre_len = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
path_training_data = 'datas/'
path_daily_run = 'datas_test/'
model_save_path = 'trained_models/'  # Directory to save the model

class AmaData(Dataset):
    def __init__(self, values, labels, scaler_path, name_symbol):
        self.values, self.labels = values, labels
        self.x_stand, self.y_stand = load_scalers(scaler_path, name_symbol)

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
        value = self.x_stand.transform(value[:, 1:])
        label = self.y_stand.transform(label[:, 1][:, None])
        value = np.float32(value)
        label = np.float32(label)
        return value, label, value_t, label_t


def load_scalers(path, name_symbol):
    x_scaler = joblib.load(os.path.join(path, f'x_scaler_{name_symbol}.pkl'))
    y_scaler = joblib.load(os.path.join(path, f'y_scaler_{name_symbol}.pkl'))
    return x_scaler, y_scaler



def read_data(name_symbol):

    # get today's date in string format "YYYY-MM-DD"
    test_date = datetime.now().strftime("%Y-%m-%d")

    # set start date to be 100 calendar days before end date
    start_date = (datetime.now() - timedelta(days=360)).strftime("%Y-%m-%d")

    # Donwload the data
    datas = yf.download(name_symbol, start=start_date, interval="1d")
    datas.reset_index(inplace=True)

    # set the Date column to be in string format
    datas['Date'] = datas['Date'].dt.strftime('%Y-%m-%d')

    # assert the data is latest
    assert datas['Date'].iloc[-1] == test_date

    # find the idx of the test_date
    test_date_idx = datas[datas['Date'] == test_date].index[0]

    # keep the s_len rows (including the test_date) and pre_len more rows
    datas = datas.iloc[test_date_idx - s_len + 1:test_date_idx + pre_len + 1]

    # Process the data
    datas = datas[["Date", "Open", "High", "Low", "Close", "Volume"]]
    datas.fillna(0, inplace=True)
    xs = datas.values[:, [1, 2, 3, 4]]
    ys = datas.values[:, 4]
    x_stand.fit(xs)
    y_stand.fit(ys[:, None])
    values, labels, labels_fake = create_data(datas)
    oos_x, oos_y, oos_y_fake = values, labels, labels_fake

    return oos_x, oos_y, oos_y_fake


def create_data(datas):
    values = []
    labels = []
    labels_fake = []
    lens = datas.shape[0]
    datas = datas.values
    for index in range(0, lens - pre_len - s_len + 1):
        value = datas[index:index + s_len, [0, 1, 2, 3, 4]]
        label = datas[index + s_len - pre_len:index + s_len + pre_len, [0, 4]]

        # create a fake label that has zeroes in the last five rows, colume 2 only
        fake_label = label.copy()
        fake_label[-pre_len:, 1] = 0
        values.append(value)
        labels.append(label)
        labels_fake.append(fake_label)

    return values, labels, labels_fake


def infer_model(name_symbol, oos_x, oos_y):

    scaler_path = path_training_data
    oos_data = AmaData(oos_x, oos_y, scaler_path, name_symbol)
    oos_data = DataLoader(oos_data, shuffle=False, batch_size=batch_size)

    model = Informer()
    model.load_state_dict(torch.load(model_save_path + f'model_{name_symbol}.pth'))
    model.to(device)

    model.eval()
    actuals, predictions = [], []
    actual_trends, predicted_trends = [], []
    actual_diffs, predicted_diffs = [], []  # Initialize lists for differences

    with torch.no_grad():
        for x, y, xt, yt in oos_data:
            mask = torch.zeros_like(y)[:, pre_len:].to(device)
            x, y, xt, yt = x.to(device), y.to(device), xt.to(device), yt.to(device)
            dec_y = torch.cat([y[:, :pre_len], mask], dim=1)
            logits = model(x, xt, dec_y, yt)

            # get prediction results
            predictions_batch = logits.cpu().numpy()
            predictions_batch_inv = y_stand.inverse_transform(predictions_batch.reshape(-1, 1)).reshape(predictions_batch.shape)

            # actuals.append(actuals_batch_inv)
            predictions.append(predictions_batch_inv)

            # Calculate the differences for binary trend prediction and differences
            for i in range(x.size(0)):

                # Last known price from the decoder initializer
                last_known_price = y[i, pre_len-1, 0].cpu().numpy()
                last_known_price_inv = y_stand.inverse_transform(last_known_price.reshape(-1, 1)).item()

                # Last predicted price
                last_predicted_price = predictions_batch_inv[i, -1, 0]

                # Last actual price from the label
                # last_actual_price = actuals_batch_inv[i, -1, 0]

                # Calculate the differences
                predicted_diff = last_predicted_price - last_known_price_inv
                # actual_diff = last_actual_price - last_known_price_inv

                # Save the actual differences and binary trends
                predicted_diffs.append(predicted_diff)
                predicted_trends.append(1 if predicted_diff > 0 else 0)

    return predictions


if __name__ == "__main__":


    # loop through the symbols_daily_run list and process each symbol
    for name_symbol in symbols_daily_run:

        # Read the data
        oos_x, oos_y, oos_y_fake = read_data(name_symbol)

        # Run inference with the pre-trained model
        oos_y_pred_with_fake_oos_y = infer_model(name_symbol, oos_x, oos_y_fake)
        # oos_y_pred_with_real_oos_y = infer_model(name_symbol, oos_x, oos_y)

        # choose which oos_y_pred to use
        oos_y_pred = oos_y_pred_with_fake_oos_y
        # oos_y_pred = oos_y_pred_with_real_oos_y

        # Flatten oos_y_pred for plotting
        oos_y_pred = np.array(oos_y_pred).flatten()

        # Select the relevant portion of oos_y
        actual_values = oos_y[0][-pre_len:, 1].astype(float)

        # add the last known price to the actual_values
        last_known_price = oos_y[0][pre_len - 1, 1]
        actual_values = np.append(last_known_price, actual_values)

        # add the last known price to the oos_y_pred
        oos_y_pred = np.append(last_known_price, oos_y_pred)

        # Calculate the differences for binary trend prediction and differences
        diffs_actual = actual_values[-1] - last_known_price
        diffs_predicted = oos_y_pred[-1] - last_known_price
        list_actual_diffs.append(diffs_actual)
        list_predicted_diffs.append(diffs_predicted)

        # Plot the actual and predicted values
        plt.figure(figsize=(10, 6))
        plt.plot(actual_values, label='Actual')
        plt.plot(oos_y_pred, label='Predicted')
        plt.xlabel('Time Step')
        plt.ylabel('Close Price')
        plt.title(f'{test_date} Actual vs. Predicted Close Price')
        plt.legend()

        # Save the figure with the date as the name postfix
        plt.savefig(f"results\\{name_symbol}_{test_date}.png")
        # plt.show()
        plt.close()


    print("Daily run complete!")