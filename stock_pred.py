import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import requests
import os
import wget

# 读取CSV文件
comp = "AMD"
url = f"https://query1.finance.yahoo.com/v7/finance/download/{comp}?period1=916963200&period2=1694304000&interval=1d&events=history&includeAdjustedClose=true"
wget.download(url, "temp.csv")
print(f"Downloaded {comp} Stock Data.")

# 读取数据并删除包含NaN的行
data = pd.read_csv('temp.csv')  # 将文件名替换为你的CSV文件名
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data.dropna(axis=0, inplace=True)  # 删除包含NaN的行

# 仅保留2020年到2024年的数据
start_date = datetime(1950, 1, 1)
end_date = datetime(2024, 12, 31)
data = data[(data.index >= start_date) & (data.index <= end_date)]

# 计算总行数并根据总行数生成 look_back_days_values
total_rows = len(data)
look_back_step = 1000
look_back_max = total_rows // 1000 * 1000

look_back_days_values = list(range(1000, look_back_max + 1, look_back_step))

# 预测未来一段时间的走势并绘制在同一个窗口中
forecast_days = 365*2
forecast_dates = [data.index[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]

num_rows = 3
num_cols = 4
num_subplots = num_rows * num_cols

# 创建第一个视窗
fig1, axes1 = plt.subplots(num_rows, num_cols, figsize=(18, 14))
fig1.tight_layout()

for idx, look_back_days in enumerate(look_back_days_values):
    row = idx // num_cols
    col = idx % num_cols

    X = []

    for i in range(len(data) - look_back_days):
        X.append(data['Close'][i:i + look_back_days].values)

    X = np.array(X)
    y = data['Close'][look_back_days:].values

    # 训练线性回归模型
    model = LinearRegression()
    model.fit(X, y)

    predicted_prices = []
    last_data = data['Close'][-look_back_days:].values.reshape(1, -1)

    for _ in range(forecast_days):
        predicted_price = model.predict(last_data)
        predicted_prices.append(predicted_price[0])
        last_data = np.roll(last_data, -1)
        last_data[0, -1] = predicted_price

    # 在子图中绘制预测结果的路径
    ax = axes1[row, col]
    ax.plot(data.index, data['Close'], label='Close')
    ax.plot(forecast_dates, predicted_prices, label=f'Look Back: {look_back_days}', linestyle='dashed')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price(USD)')
    ax.set_title(f'{comp} Stock Prediction (Look Back: {look_back_days})')
    ax.legend()

plt.subplots_adjust(hspace=0.5)

# 创建第二个视窗
fig2, ax2 = plt.subplots(figsize=(10, 6))

# 计算平均值并绘制
average_predicted_prices = np.mean(np.array([ax.lines[line_idx * 2 + 1].get_ydata() for line_idx in range(len(look_back_days_values)) if len(ax.lines) > line_idx * 2 + 1]), axis=0)

# 绘制前一年的数据（蓝色实线）
one_year_ago = data.index[-forecast_days:]
one_year_ago_prices = data['Close'][-forecast_days:]

ax2.plot(one_year_ago, one_year_ago_prices, label='One Year Ago', linestyle='solid')
ax2.plot(forecast_dates, average_predicted_prices, label='Average Predicted Price', linestyle='dashed')
ax2.set_xlabel('Date')
ax2.set_ylabel('Price(USD)')
ax2.set_title(f'Average {comp} Stock Prediction')
ax2.legend()

# 显示第二个视窗
plt.show()
os.remove("temp.csv")
