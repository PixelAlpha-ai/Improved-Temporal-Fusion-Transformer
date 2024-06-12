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
import seaborn as sns
import os
import joblib
from sklearn.metrics import confusion_matrix, classification_report

# Define global parameters
train_size = 0.7  # Proportion of data to use for training
test_size = 0.15  # Proportion of data to use for the OOS test set
val_size = 0.15  # Proportion of data to use for validation
x_stand = StandardScaler()
y_stand = StandardScaler()
s_len = 64
pre_len = 5
batch_size = 32
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 5e-5
epochs = 100
patience = 5  # Early stopping patience
path_training_data = 'datas/'
path_results_summary = 'results_summary/'
model_save_path = 'trained_models/'  # Directory to save the model


def save_scalers(x_scaler, y_scaler, name_symbol, path):
    joblib.dump(x_scaler, os.path.join(path, f'x_scaler_{name_symbol}.pkl'))
    joblib.dump(y_scaler, os.path.join(path, f'y_scaler_{name_symbol}.pkl'))

def load_scalers(path, name_symbol):
    x_scaler = joblib.load(os.path.join(path, f'x_scaler_{name_symbol}.pkl'))
    y_scaler = joblib.load(os.path.join(path, f'y_scaler_{name_symbol}.pkl'))
    return x_scaler, y_scaler


def save_oos_data(oos_x, oos_y, name_symbol, save_directory):

    # Create the directory if it does not exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Iterate through each sample in the OOS data
    for i, (sample_x, sample_y) in enumerate(zip(oos_x, oos_y)):
        # Convert the sample features (numpy array) to a DataFrame
        sample_x_df = pd.DataFrame(sample_x, columns=['Open', 'High', 'Low', 'Close', 'Volume'])

        # Convert the sample labels (numpy array) to a DataFrame
        sample_y_df = pd.DataFrame(sample_y,
                                   columns=['Open', 'Close'])  # Adjust columns based on your specific structure

        # Save the DataFrame to a CSV file
        # file_name_x = os.path.join(save_directory, f"oos_x_{name_symbol}_{i}.csv")
        # sample_x_df.to_csv(file_name_x, index=False)
        #
        # file_name_y = os.path.join(save_directory, f"oos_y_{name_symbol}_{i}.csv")
        # sample_y_df.to_csv(file_name_y, index=False)
        # print(f"Saved: {file_name_x}")

def create_data(datas):
    values = []
    labels = []
    lens = datas.shape[0]
    datas = datas.values
    for index in range(0, lens - pre_len - s_len):
        value = datas[index:index + s_len, [0, 1, 2, 3, 4]]
        label = datas[index + s_len - pre_len:index + s_len + pre_len, [0, 4]]
        values.append(value)
        labels.append(label)
    return values, labels

def read_data(name_symbol):

    datas = pd.read_csv(f"{path_training_data}\\{name_symbol}.csv")

    # Process and split data as before
    datas = datas[["Date", "Open", "High", "Low", "Close", "Volume"]]
    datas.fillna(0, inplace=True)
    xs = datas.values[:, [1, 2, 3, 4]]
    ys = datas.values[:, 4]
    x_stand.fit(xs)
    y_stand.fit(ys[:, None])
    values, labels = create_data(datas)

    # Save the scalers
    save_scalers(x_stand, y_stand, name_symbol, path_training_data)

    # Split off the OOS test set first
    train_val_x, oos_x, train_val_y, oos_y = train_test_split(values, labels, test_size=test_size)

    # Split the remaining data into train and validation sets
    train_x, val_x, train_y, val_y = train_test_split(train_val_x, train_val_y, train_size=train_size/(train_size + val_size))

    # Define the directory to save the OOS data
    oos_save_directory = os.path.join(path_training_data, 'oos_data')

    # Save OOS data to the new directory
    # save_oos_data(oos_x, oos_y, name_symbol, oos_save_directory)

    return train_x, val_x, train_y, val_y, oos_x, oos_y

class AmaData(Dataset):
    def __init__(self, values, labels, name_symbol, scaler_path):
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

def train_model(name_symbol, train_x, train_y, val_x, val_y):

    scaler_path = path_training_data
    train_data = AmaData(train_x, train_y, name_symbol=name_symbol, scaler_path=scaler_path)
    train_data = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_data = AmaData(val_x, val_y, name_symbol=name_symbol, scaler_path=scaler_path)
    val_data = DataLoader(val_data, shuffle=True, batch_size=batch_size)

    model = Informer()
    model.to(device)
    loss_fc = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience_counter = 0

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

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the model if it has the best validation loss so far
            save_path = model_save_path + f'model_{name_symbol}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Training Complete.")

def infer_model(name_symbol, oos_x, oos_y):

    scaler_path = path_training_data
    oos_data = AmaData(oos_x, oos_y, name_symbol=name_symbol, scaler_path=scaler_path)
    oos_data = DataLoader(oos_data, shuffle=False, batch_size=batch_size)

    model = Informer()
    model.load_state_dict(torch.load(model_save_path + f'model_{name_symbol}.pth'))
    model.to(device)

    loss_fc = nn.MSELoss()
    model.eval()
    oos_loss = 0.0
    actuals, predictions = [], []
    actual_trends, predicted_trends = [], []
    actual_diffs, predicted_diffs = [], []  # Initialize lists for differences

    with torch.no_grad():
        for x, y, xt, yt in oos_data:
            mask = torch.zeros_like(y)[:, pre_len:].to(device)
            x, y, xt, yt = x.to(device), y.to(device), xt.to(device), yt.to(device)
            dec_y = torch.cat([y[:, :pre_len], mask], dim=1)
            logits = model(x, xt, dec_y, yt)
            loss = loss_fc(logits, y[:, pre_len:])
            oos_loss += loss.item()

            # Inverse transform the normalized values back to the original scale
            actuals_batch = y[:, pre_len:].cpu().numpy()
            predictions_batch = logits.cpu().numpy()
            actuals_batch_inv = y_stand.inverse_transform(actuals_batch.reshape(-1, 1)).reshape(actuals_batch.shape)
            predictions_batch_inv = y_stand.inverse_transform(predictions_batch.reshape(-1, 1)).reshape(predictions_batch.shape)

            actuals.append(actuals_batch_inv)
            predictions.append(predictions_batch_inv)

            # Calculate the differences for binary trend prediction and differences
            for i in range(x.size(0)):

                # Last known price from the decoder initializer
                last_known_price = y[i, pre_len-1, 0].cpu().numpy()
                last_known_price_inv = y_stand.inverse_transform(last_known_price.reshape(-1, 1)).item()

                # Last predicted price
                last_predicted_price = predictions_batch_inv[i, -1, 0]

                # Last actual price from the label
                last_actual_price = actuals_batch_inv[i, -1, 0]

                # Calculate the differences
                predicted_diff = last_predicted_price - last_known_price_inv
                actual_diff = last_actual_price - last_known_price_inv

                # Save the actual differences and binary trends
                actual_diffs.append(actual_diff)
                predicted_diffs.append(predicted_diff)
                actual_trends.append(1 if actual_diff > 0 else 0)
                predicted_trends.append(1 if predicted_diff > 0 else 0)

    oos_loss /= len(oos_data)
    print(f"OOS Test Loss: {oos_loss}")

    # Scatter plot for the differences
    plt.figure(figsize=(10, 6))
    plt.scatter(actual_diffs, predicted_diffs)
    # calcualte R2
    r2 = np.corrcoef(actual_diffs, predicted_diffs)[0, 1]
    plt.xlabel('Actual Price Difference')
    plt.ylabel('Predicted Price Difference')
    plt.title('Scatter Plot of Actual vs. Predicted Price Differences')
    plt.savefig(f"{path_results_summary}\\Diff_{name_symbol}_R2{r2}.png")

    # Confusion matrix for the trend prediction
    cm = confusion_matrix(actual_trends, predicted_trends)
    cm_df = pd.DataFrame(cm, index=['Down', 'Up'], columns=['Predicted Down', 'Predicted Up'])
    accuracy = np.trace(cm) / np.sum(cm)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix for Trend Prediction')
    plt.savefig(f"{path_results_summary}\\CM_{name_symbol}_ACC{accuracy}.png")

    # Print classification report
    print(classification_report(actual_trends, predicted_trends, target_names=['Down', 'Up']))


if __name__ == '__main__':

    # Import the list of symbols for US stock and crypto
    from config import us_stock_symbols, crypto_symbols, symbols_daily_run

    # Train the US stocks
    for name_symbol in crypto_symbols:

        # Read the data
        train_x, val_x, train_y, val_y, oos_x, oos_y = read_data(name_symbol)

        # Uncomment the next line to train the model
        train_model(name_symbol, train_x, train_y, val_x, val_y)

        # Run inference with the pre-trained model
        infer_model(name_symbol, oos_x, oos_y)

        # # Run inference on the in-sample data
        # infer_model(name_symbol, train_x, train_y)
        #
        # # Run inference on the in-sample data
        # infer_model(name_symbol, val_x, val_y)


