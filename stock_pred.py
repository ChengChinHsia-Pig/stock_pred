import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# 读取CSV文件
comp = "NVDA"
data = pd.read_csv(f'{comp}.csv')  # 将文件名替换为你的CSV文件名
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# 仅保留2020年到2024年的数据
start_date = datetime(1980, 3, 17)
end_date = datetime(2024, 12, 31)
data = data[(data.index >= start_date) & (data.index <= end_date)]

# 不同的 look_back_days 值
look_back_days_values = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000]
# look_back_days_values = list(range(1, 6001))

# 预测未来一段时间的走势并绘制在同一个窗口中
forecast_days = 365
forecast_dates = [data.index[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]

num_rows = 5
num_cols = 6
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

# 显示第一个视窗，但不阻塞程序
plt.show(block=False)

# 创建第二个视窗并显示
fig2, ax2 = plt.subplots(figsize=(10, 6))
# 计算平均值并绘制
average_predicted_prices = np.mean(np.array([ax.lines[line_idx * 2 + 1].get_ydata() for line_idx in range(len(look_back_days_values)) if len(ax.lines) > line_idx * 2 + 1]), axis=0)
ax2.plot(forecast_dates, average_predicted_prices, label='Average Predicted Price', linestyle='dashed', color='orange')
ax2.set_xlabel('Date')
ax2.set_ylabel('Price(USD)')
ax2.set_title(f'Average {comp} Stock Prediction')
ax2.legend()

# 显示第二个视窗
plt.show()
