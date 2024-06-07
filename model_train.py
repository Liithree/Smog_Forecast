import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size), torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


# 读取数据
file_name = 'output.csv'
df = pd.read_csv(file_name)
features = df[
    ['rh', 'wind_spd', 'slp', 'azimuth', 'dewpt', 'snow', 'wind_dir', 'code', 'vis', 'precip', 'pres', 'temp']]
target = df['aqi']

# 标准化
scaler = StandardScaler()
features = scaler.fit_transform(features)
print(features)

# 创建时间序列数据集
def create_sequences(features, target, seq_length):
    xs, ys = [], []
    for i in range(len(features) - seq_length):
        x = features[i:i + seq_length]
        y = target[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


seq_length = 2
X, y = create_sequences(features, target, seq_length)

# 打印数据的形状
print(f'X shape: {X.shape}')
print(f'y shape: {y.shape}')

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

input_size = features.shape[1]
hidden_layer_size = 50
output_size = 1

model = LSTMModel(input_size, hidden_layer_size, output_size)

# 训练模型
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 50

for i in range(epochs):
    for seq, labels in zip(X_train, y_train):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size), torch.zeros(1, 1, hidden_layer_size))

        y_pred = model(torch.FloatTensor(seq))

        single_loss = loss_function(y_pred, torch.FloatTensor([labels]))
        single_loss.backward()
        optimizer.step()
    print(f'Epoch {i} loss: {single_loss.item()}')

print(f'Final loss: {single_loss.item()}')

# 评估模型
test_predictions = []
model.eval()
for seq in X_test:
    with torch.no_grad():
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size), torch.zeros(1, 1, hidden_layer_size))
        test_predictions.append(model(torch.FloatTensor(seq)).item())

# 转换为numpy数组
test_predictions = np.array(test_predictions)
# 均方根误差
mse = np.mean((test_predictions - y_test) ** 2)
print(f'Test MSE: {mse}')
