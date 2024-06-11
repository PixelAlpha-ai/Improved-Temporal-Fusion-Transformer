"""
Main function for daily run
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import os
import joblib
import yfinance as yf
from models import Informer
from config import symbols_daily_run
from datetime import datetime, timedelta
import plotly.graph_objects as go

from pandas.tseries.holiday import USFederalHolidayCalendar


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


def read_data(name_symbol, test_date):

    # convert the test_date to datetime object
    test_date_dt = datetime.strptime(test_date, "%Y-%m-%d")

    # set start date to be 100 calendar days before test date
    start_date = (test_date_dt - timedelta(days=360)).strftime("%Y-%m-%d")

    # Donwload the data
    # datas = yf.download(name_symbol, start=start_date, interval="1d")
    # datas.reset_index(inplace=True)
    # set the Date column to be in string format
    # datas['Date'] = datas['Date'].dt.strftime('%Y-%m-%d')

    # temp - if no access to internet
    datas = pd.read_csv(f'datas_test\\{name_symbol}.csv')

    # assert the data is latest
    # assert datas['Date'].iloc[-1] == test_date

    # find the idx of the test_date
    test_date_idx = datas[datas['Date'] == test_date].index[0]

    # keep the s_len rows (including the test_date) and pre_len more rows
    # datas = datas.iloc[test_date_idx - s_len + 1:test_date_idx + pre_len + 1]
    datas = datas.iloc[test_date_idx - s_len - pre_len + 1: test_date_idx + 1]

    # Process the data
    datas = datas[["Date", "Open", "High", "Low", "Close", "Volume"]]
    # datas.fillna(0, inplace=True)
    xs = datas.values[:, [1, 2, 3, 4]]
    ys = datas.values[:, 4]
    x_stand.fit(xs)
    y_stand.fit(ys[:, None])
    values, labels_fake = create_data_no_label(datas)
    oos_x, oos_y_fake = values, labels_fake

    return oos_x, oos_y_fake


def create_data_no_label(datas):
    values = []
    labels = []
    labels_fake = []
    lens = datas.shape[0]
    datas = datas.values

    # for index in range(0, lens - s_len + 1):
    # only run the last one valid data
    for index in [lens - s_len]:

        # get the input value, 64 rows up to today
        value = datas[index:index + s_len, [0, 1, 2, 3, 4]]

        # get the label pt1, 5 rows for the last five days including today
        label_known = datas[index + s_len - pre_len:index + s_len, [0, 4]]

        # generate future trading days starting from the last known date
        last_date = pd.to_datetime(label_known[-1, 0])

        # Define custom business day with US Federal holidays
        us_bd = pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar())

        future_dates = [last_date + i * us_bd for i in range(1, pre_len + 1)]

        # create the label pt 2, with future dates and Close price as zero
        label_fake = np.zeros((pre_len, 2), dtype=object)
        for i, future_date in enumerate(future_dates):
            label_fake[i, 0] = future_date.strftime('%Y-%m-%d')
            label_fake[i, 1] = 0  # Close price is set to zero

        # merge the label pt1 and pt2 to generate the fake_label
        fake_label = np.vstack([label_known, label_fake])

        # append the values, labels and fake_labels to the list
        values.append(value)
        labels_fake.append(fake_label)

    return values, labels_fake


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

                # Calculate the differences
                predicted_diff = last_predicted_price - last_known_price_inv

                # Save the actual differences and binary trends
                predicted_diffs.append(predicted_diff)
                predicted_trends.append(1 if predicted_diff > 0 else 0)

    return predictions


if __name__ == "__main__":

    # get today's date in string format "YYYY-MM-DD"
    test_date = datetime.now().strftime("%Y-%m-%d")
    test_date = '2024-02-21'

    # loop through the symbols_daily_run list and process each symbol
    for name_symbol in symbols_daily_run:

        print(f"Processing {name_symbol}...")

        # Read the data
        oos_x, oos_y_fake = read_data(name_symbol, test_date)

        # Run inference with the pre-trained model
        oos_y_pred = infer_model(name_symbol, oos_x, oos_y_fake)

        # Flatten oos_y_pred for plotting
        oos_y_pred = np.array(oos_y_pred).flatten()

        # assign the predicted values to the last pre_l rows of the oos_y_fake
        oos_y_fake[-1][-pre_len:, 1] = oos_y_pred

        # convert data to dataframe
        df_oos_x = pd.DataFrame(oos_x[0])
        df_oos_x.columns = ['Date', 'Open', 'High', 'Low', 'Close']
        df_oos_x['Date'] = pd.to_datetime(df_oos_x['Date'])
        df_oos_x[['Open', 'High', 'Low', 'Close']] = df_oos_x[['Open', 'High', 'Low', 'Close']].astype(float)

        df_oos_y = pd.DataFrame(oos_y_fake[0][-pre_len:])
        df_oos_y.columns = ['Date', 'Close']
        df_oos_y['Date'] = pd.to_datetime(df_oos_y['Date'])
        df_oos_y['Close'] = df_oos_y['Close'].astype(float)

        # Plot OHLC candlestick chart for df_oos_x
        fig = go.Figure(data=[go.Candlestick(x=df_oos_x['Date'],
                                             open=df_oos_x['Open'],
                                             high=df_oos_x['High'],
                                             low=df_oos_x['Low'],
                                             close=df_oos_x['Close'],
                                             name='OHLC')])

        # Plot line chart for df_oos_y
        fig.add_trace(go.Scatter(x=df_oos_y['Date'], y=df_oos_y['Close'], mode='lines', name='Predicted Close'))

        # Customize layout
        fig.update_layout(title='',
                          xaxis_title='',
                          yaxis_title='Price',
                          xaxis_rangeslider_visible=False,
                          showlegend=False)

        # Add a vertical line at the test_date
        fig.add_vline(x=test_date, line_dash="dash", line_color="black")

        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))

        # save the iamge as a png file
        fig.write_image(f"output/{name_symbol}_{test_date}.png")
