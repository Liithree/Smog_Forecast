import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 读取CSV文件
data = pd.read_csv('../output1.csv')

features = ['rh', 'wind_spd', 'wind_dir', 'vis', 'pres', 'temp']
target = 'aqi'

# 数据预处理
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# 构建特征和标签
X = data[features].values
y = data[target].values


# 创建时间序列数据
def create_sequences(X, y, seq_length):
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:i + seq_length])
        ys.append(y[i + seq_length])
    return np.array(xs), np.array(ys)


seq_length = 24  # 使用过去24小时的数据预测未来的AQI值
X_seq, y_seq = create_sequences(X, y, seq_length)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)


# 创建数据集和数据加载器
class AirQualityDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = AirQualityDataset(X_train, y_train)
test_dataset = AirQualityDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 定义更加复杂的LSTM模型用于回归
class AirQualityLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5):
        super(AirQualityLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.batch_norm1(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.batch_norm2(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out


input_size = len(features)
hidden_size = 256
num_layers = 3

model = AirQualityLSTM(input_size, hidden_size, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')



# 模型评估
model.eval()
total_loss = 0
all_labels = []
all_predictions = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(outputs.cpu().numpy())

average_loss = total_loss / len(test_loader)
print(f'Test Average Loss: {average_loss:.4f}')
rmse = np.sqrt(average_loss)
print(f'Test RMSE: {rmse:.4f}')

torch.save(model, '../model_LSTM_r_complex2.pth')

results = pd.DataFrame({
    'Actual AQI': all_labels,
    'Predicted AQI': all_predictions
})
results.to_csv('result1.csv', index=False)

