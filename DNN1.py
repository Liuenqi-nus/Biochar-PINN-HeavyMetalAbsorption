import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import random
import matplotlib.pyplot as plt
import logging
from datetime import datetime
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# 设置日志记录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f'DNN_{timestamp}.log'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

# 加载数据集
train_df = pd.read_csv('Data7/train.csv')
valid_df = pd.read_csv('Data7/valid.csv')
test_df = pd.read_csv('Data7/test.csv')

# 数据预处理
# columns_to_exclude = ['biomass name', 'adsorption efficiency (%)', 'Ct','M1', 'M2', 'M3']
columns_to_scale100 = [ 'n(Fe3+):m(BC)', 'n(Fe2+):m(BC)',
                       'n(Ag+):m(BC)', 'n(Al3+):m(BC)', 'n(Ce3+):m(BC)', 'n(Cu2+):m(BC)', 'n(La3+):m(BC)', 'n(Mn2+):m(BC)', 'n(Mn7+):m(BC)', 'n(Mg2+):m(BC)', 'n(Zn2+):m(BC)']


def preprocess_data(df):
    for col in df.columns:
        if col in columns_to_scale100:
            df[col] *= 10000  # 扩大100倍
        if df[col].dtype == 'float64':
            df[col] = df[col].round(2)  # 将数值型列的精度修改为2位小数

    # 应用标签编码
    label_encoder = LabelEncoder()
    categorical_columns = ['biomass type', 'methods',  'metal ions']
    for col in categorical_columns:
        df[col] = label_encoder.fit_transform(df[col])
    return df


train_df = preprocess_data(train_df)
valid_df = preprocess_data(valid_df)
test_df = preprocess_data(test_df)

# 分离特征和目标变量
X_train = train_df.drop( ['adsorption capacity/mg·g-1'], axis=1).values
y_train = train_df['adsorption capacity/mg·g-1'].values

X_valid = valid_df.drop( ['adsorption capacity/mg·g-1'], axis=1).values
y_valid = valid_df['adsorption capacity/mg·g-1'].values

X_test = test_df.drop(['adsorption capacity/mg·g-1'], axis=1).values
y_test = test_df['adsorption capacity/mg·g-1'].values

# 转换为torch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)


# 定义模型
class RegressionModel(nn.Module):
    def __init__(self, n_features):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(n_features, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, 128)
        self.fc8 = nn.Linear(128, 128)
        self.fc9 = nn.Linear(128, 128)
        self.output = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc7(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc8(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc9(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.output(x)
        return x


# 实例化模型
model = RegressionModel(X_train.shape[1])
optimizer = Adam(model.parameters(), lr=0.0001)  # 降低学习率
criterion = nn.MSELoss()

num_epochs = 10000
train_losses = []
validation_losses = []
feature_contributions = []  # 存储特征贡献

for epoch in range(num_epochs):
    model.train()
    batch_losses = []
    epoch_gradients = []
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs.requires_grad = True
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        batch_losses.append(loss.item())

        # 记录特征贡献
        gradients = inputs.grad.detach().abs().mean(dim=0)
        epoch_gradients.append(gradients)

    # 添加本次epoch的平均特征贡献
    train_loss = np.mean(batch_losses)
    train_losses.append(train_loss)

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_valid_tensor)
        val_loss = criterion(val_outputs.squeeze(), y_valid_tensor)
        validation_losses.append(val_loss.item())

    if (epoch + 1) % 10 == 0:
        logging.info(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # 打印训练损失
        print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # 打印5个随机预测值及其真实值
        sample_indices = random.sample(range(len(y_valid_tensor)), 5)
        sample_predictions = val_outputs[sample_indices].detach().numpy()
        sample_targets = y_valid_tensor[sample_indices].numpy()
        for i in range(5):
            print(f'Sample Prediction: {sample_predictions[i]}, Actual Value: {sample_targets[i]}')


# 定义 MAPE 计算函数
def mean_absolute_percentage_error_filtered(y_true, y_pred):
    mask = y_true > 1e-2  # 只考虑真实值大于0.01的情况
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    if len(y_true_filtered) > 0:
        return torch.mean(torch.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
    else:
        return torch.tensor(0.0)  # 如果过滤后没有数据，返回0


# 计算 MSE, MAPE, MAE 和 R²
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    mse = criterion(y_test_tensor, test_outputs.squeeze())
    mape = mean_absolute_percentage_error_filtered(y_test_tensor, test_outputs.squeeze())
    mae = mean_absolute_error(y_test_tensor.numpy(), test_outputs.squeeze().numpy())
    r2 = r2_score(y_test_tensor.numpy(), test_outputs.squeeze().numpy())

logging.info(f'Final Mean Squared Error: {mse.item()}')
logging.info(f'Mean Absolute Percentage Error: {mape.item()}%')
logging.info(f'Mean Absolute Error (MAE): {mae}')
logging.info(f'R² Score: {r2}')

# 打印评估结果
print(f'Final Mean Squared Error: {mse.item()}')
print(f'Mean Absolute Percentage Error: {mape.item()}%')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R² Score: {r2}')

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
loss_plot_filename = f'DNN_loss_plot_{timestamp}.png'
plt.savefig(loss_plot_filename)
plt.show()
