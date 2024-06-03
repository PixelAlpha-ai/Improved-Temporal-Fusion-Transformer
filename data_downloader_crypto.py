import yfinance as yf
import os
from config import *

# Define the start and end dates
# start_date = "2010-01-01"
# end_date = "2023-12-31"
# output_dir = "datas"

start_date = "2024-03-01"
end_date = "2024-05-31"
output_dir = "datas_test"

def download_daily_data(symbol):
    try:
        data = yf.download(symbol, start=start_date, end=end_date, interval="1d")
        if not data.empty:
            data.to_csv(os.path.join(output_dir, f"{symbol}.csv"))
            print(f"Downloaded data for {symbol}")
        else:
            print(f"No data downloaded for {symbol}")
    except Exception as e:
        print(f"Failed to download data for {symbol}: {e}")

# Download the data for each stock
for symbol in crypto_symbols:
    download_daily_data(symbol)

print("Data download complete!")