# Improved Temporal Fusion Transformer
This is the project folder submitted by team PixelAlpha.ai for the 2024 Franklin Templeton AI contest.
The contents in this project folder are:

Folders:
- Datas : Folder hosting training data for model R&D (Train/Val/Testing)
- Datas_test: Folder hosting real-time deployment data (updated daily)
- Models: Folder hosting all core AI codes.
- Output: Location where real-time deployment results are saved. These are images used in the Frontend.
- Results_summary: Results generated during the model training phase.
- Trained models: Location where trained models are saved

Files:
config.py : general configuration file
data_downlolader_crypto.py: download training or testing cryptocurrency data from Yahoo finance for given time range and symbol list.
data_downlolader_stock.py: download training or testing US stock data from Yahoo finance for given time range and symbol list.
holdout_test_stock: apply the trained model on unseen held-out extra data.
main.py: main code for daily run, to apply trained model to predict daily price changes/
train_stock_model_validation.py: main code for the model training and validation and testing.
