import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import requests
import os
import wget
import time
import concurrent.futures

unix = round(time.time())
comp = "AAPL"
url = f"https://query1.finance.yahoo.com/v7/finance/download/{comp}?period1=000000000&period2={unix}&interval=1d&events=history&includeAdjustedClose=true"
wget.download(url, "temp.csv")
print(f"Downloaded {comp} Stock Data.")

start_time = time.time()
data = pd.read_csv('temp.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data.dropna(axis=0, inplace=True)

start_date = datetime(1950, 1, 1)
end_date = datetime(2024, 12, 31)
data = data[(data.index >= start_date) & (data.index <= end_date)]

total_rows = len(data)
look_back_step = 1000
look_back_max = total_rows // 1000 * 1000
look_back_days_values = list(range(1000, look_back_max + 1, look_back_step))
forecast_days = 365
forecast_dates = [data.index[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]

num_rows = 3
num_cols = 4
num_subplots = num_rows * num_cols
fig1, axes1 = plt.subplots(num_rows, num_cols, figsize=(18, 14))
fig1.tight_layout()
def train_and_predict(look_back_days):
    X = []

    for i in range(len(data) - look_back_days):
        X.append(data['Close'][i:i + look_back_days].values)

    X = np.array(X)
    y = data['Close'][look_back_days:].values

    model = LinearRegression()
    model.fit(X, y)

    predicted_prices = []
    last_data = data['Close'][-look_back_days:].values.reshape(1, -1)

    for _ in range(forecast_days):
        predicted_price = model.predict(last_data)
        predicted_prices.append(predicted_price[0])
        last_data = np.roll(last_data, -1)
        last_data[0, -1] = predicted_price

    return look_back_days, predicted_prices

num_threads = 64
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    results = list(executor.map(train_and_predict, look_back_days_values))

for idx, (look_back_days, predicted_prices) in enumerate(results):
    row = idx // num_cols
    col = idx % num_cols
    ax = axes1[row, col]
    ax.plot(data.index, data['Close'], label='Close')
    ax.plot(forecast_dates, predicted_prices, label=f'Look Back: {look_back_days}', linestyle='dashed')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price(USD)')
    ax.set_title(f'{comp} Stock Prediction (Look Back: {look_back_days})')
    ax.legend()

plt.subplots_adjust(hspace=0.5)
fig2, ax2 = plt.subplots(figsize=(10, 6))
average_predicted_prices = np.mean(np.array([ax.lines[line_idx * 2 + 1].get_ydata() for line_idx in range(len(look_back_days_values)) if len(ax.lines) > line_idx * 2 + 1]), axis=0)
one_year_ago = data.index[-forecast_days:]
one_year_ago_prices = data['Close'][-forecast_days:]

try:
    ax2.plot(one_year_ago, one_year_ago_prices, label=f'{forecast_days} Days Ago', linestyle='solid')
    ax2.plot(forecast_dates, average_predicted_prices, label='Average Predicted Price', linestyle='dashed')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Price(USD)')
    ax2.set_title(f'Average {comp} Stock Prediction')
    ax2.legend()
except ValueError as e:
    print("資料錯誤:", e)
    os.remove("temp.csv")

total_time = time.time()-start_time
print(f"Total time: {total_time} s")
plt.show()
os.remove("temp.csv")
