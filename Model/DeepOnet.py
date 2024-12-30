import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime
import pickle
import time
import json

torch.manual_seed(11)
np.random.seed(11)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(11)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


with open('metal_ion_features.json', 'r') as f:
    metal_ion_features = json.load(f)


model_save_path = 'data8/deeponet_model.pth'
train_pred_save_path = 'data8/train_predictions.pt'
val_pred_save_path = 'data8/val_predictions.pt'
test_pred_save_path = 'data8/test_predictions.pt'

train_file_path = 'data9/train.csv'
valid_file_path = 'data9/valid.csv'
test_file_path = 'data9/test1.csv'

train_data = pd.read_csv(train_file_path)
valid_data = pd.read_csv(valid_file_path)
test_data = pd.read_csv(test_file_path)

def encode_text_features(data, label_encoders=None, is_train=True):

    text_features = data[['atmosphere', 'biomass type', 'methods', 'metal ions']]
    encoded_text_features = []

    if is_train:
        label_encoders = {}
        for col in text_features.columns[:-1]: 
            le = LabelEncoder()
            encoded_col = le.fit_transform(text_features[col])
            label_encoders[col] = le
            encoded_text_features.append(encoded_col)
    else:
        for col in text_features.columns[:-1]:
            encoded_col = label_encoders[col].transform(text_features[col])
            encoded_text_features.append(encoded_col)

   
    metal_feature_data = []
    for idx, metal in text_features['metal ions'].items():
        if metal in metal_ion_features:
            metal_content = 1   
            metal_features = metal_ion_features[metal]  
            metal_feature_data.append([metal_content] + metal_features)
        else:
            metal_feature_data.append([0] * 7) 

   
    encoded_text_features = np.column_stack(encoded_text_features)
    metal_feature_data = np.array(metal_feature_data)
    encoded_text_features = np.hstack((encoded_text_features, metal_feature_data))

    feature_names = ['content', 'radius', 'electronegativity', 'max_oxidation_state', 'hydration_energy', 'group', 'period']
    metal_feature_df = pd.DataFrame(metal_feature_data, columns=[f'metal_{feature_name}' for feature_name in feature_names], index=data.index)
    data = pd.concat([data, metal_feature_df], axis=1)

    return encoded_text_features, label_encoders, data

train_encoded_text_features, label_encoders, train_data = encode_text_features(train_data, is_train=True)
val_encoded_text_features, _, valid_data = encode_text_features(valid_data, label_encoders, is_train=False)
test_encoded_text_features, _, test_data = encode_text_features(test_data, label_encoders, is_train=False)

branch_columns = [' T1', 'T2', 'T3', 'pH', 'time1', 'time2', 'Fe3', 'Fe2', 'Ag', 'Al3', 'Ce3', 'Cu2', 'La3', 'Mn2',
                  'Mn7', 'Mg2', 'Zn2', 'BC', 'C0', 'mg'] 
branch_columns.extend([f'metal_{feature_name}' for feature_name in ['content', 'radius', 'electronegativity', 'max_oxidation_state', 'hydration_energy', 'group', 'period']])

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

trunk_columns = ['time3']

branch_inputs = train_data[branch_columns].values
trunk_inputs = train_data[trunk_columns].values
outputs = train_data[['qt']].values

valid_branch_inputs = valid_data[branch_columns].values
valid_trunk_inputs = valid_data[trunk_columns].values
valid_outputs = valid_data[['qt']].values

test_branch_inputs = test_data[branch_columns].values
test_trunk_inputs = test_data[trunk_columns].values
test_outputs = test_data[['qt']].values

branch_scaler = StandardScaler()
trunk_scaler = StandardScaler()

branch_inputs_scaled = branch_scaler.fit_transform(branch_inputs)
trunk_inputs_scaled = trunk_scaler.fit_transform(trunk_inputs)

valid_branch_inputs_scaled = branch_scaler.transform(valid_branch_inputs)
valid_trunk_inputs_scaled = trunk_scaler.transform(valid_trunk_inputs)

test_branch_inputs_scaled = branch_scaler.transform(test_branch_inputs)
test_trunk_inputs_scaled = trunk_scaler.transform(test_trunk_inputs)

with open('data8/branch_scaler.pkl', 'wb') as f:
    pickle.dump(branch_scaler, f)

with open('data8/trunk_scaler.pkl', 'wb') as f:
    pickle.dump(trunk_scaler, f)

branch_inputs_tensor = torch.tensor(branch_inputs_scaled, dtype=torch.float32).to(device)
trunk_inputs_tensor = torch.tensor(trunk_inputs_scaled, dtype=torch.float32).to(device)
outputs_tensor = torch.tensor(outputs, dtype=torch.float32).to(device)

val_branch_inputs_tensor = torch.tensor(valid_branch_inputs_scaled, dtype=torch.float32).to(device)
val_trunk_inputs_tensor = torch.tensor(valid_trunk_inputs_scaled, dtype=torch.float32).to(device)
val_outputs_tensor = torch.tensor(valid_outputs, dtype=torch.float32).to(device)

test_branch_inputs_tensor = torch.tensor(test_branch_inputs_scaled, dtype=torch.float32).to(device)
test_trunk_inputs_tensor = torch.tensor(test_trunk_inputs_scaled, dtype=torch.float32).to(device)
test_outputs_tensor = torch.tensor(test_outputs, dtype=torch.float32).to(device)

batch_size = 16
train_dataset = TensorDataset(branch_inputs_tensor, trunk_inputs_tensor, outputs_tensor)
valid_dataset = TensorDataset(val_branch_inputs_tensor, val_trunk_inputs_tensor, val_outputs_tensor)
test_dataset = TensorDataset(test_branch_inputs_tensor, test_trunk_inputs_tensor, test_outputs_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class BranchNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BranchNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size //4)
        self.fc4 = nn.Linear(hidden_size //4, hidden_size // 8)
        self.fc5 = nn.Linear(hidden_size //8, output_size)
        self.relu = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class TrunkNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TrunkNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc4 = nn.Linear(hidden_size // 4, hidden_size // 8)
        self.fc5 = nn.Linear(hidden_size // 8, output_size)
        self.relu = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class DeepONet(nn.Module):
    def __init__(self, branch_input_size, trunk_input_size, hidden_size, output_size):
        super(DeepONet, self).__init__()
        self.branch_net = BranchNetwork(branch_input_size, hidden_size, output_size)
        self.trunk_net = TrunkNetwork(trunk_input_size, hidden_size, output_size)

    def forward(self, branch_input, trunk_input):
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        combined_output = torch.mul(branch_output, trunk_output)
        qt_prediction = combined_output.sum(dim=1, keepdim=True)
        return qt_prediction

branch_input_size = branch_inputs.shape[1]
trunk_input_size = trunk_inputs.shape[1]
hidden_size = 64
output_size = 1  # 只预测 qt

deeponet_model = DeepONet(branch_input_size, trunk_input_size, hidden_size, output_size).to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(deeponet_model.parameters(), lr=0.0005)

epochs = 2000
train_r2_scores = []
train_mse_scores = []
total_losses = []
valid_losses = []

for epoch in range(epochs):
    deeponet_model.train()
    total_loss_epoch = 0.0

    for batch_idx, (branch_batch, trunk_batch, output_batch) in enumerate(train_loader):
        branch_batch = branch_batch.to(device)
        trunk_batch = trunk_batch.to(device)
        output_batch = output_batch.to(device)

        qt_predictions = deeponet_model(branch_batch, trunk_batch)

        loss = criterion(qt_predictions, output_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss_epoch += loss.item()

    total_losses.append(total_loss_epoch / len(train_loader))

    if (epoch + 1) % 10 == 0:
        deeponet_model.eval()
        with torch.no_grad():
            all_train_predictions = []
            all_train_outputs = []

            for branch_batch, trunk_batch, output_batch in train_loader:
                branch_batch = branch_batch.to(device)
                trunk_batch = trunk_batch.to(device)
                output_batch = output_batch.to(device)

                qt_predictions = deeponet_model(branch_batch, trunk_batch)
                all_train_predictions.append(qt_predictions.cpu().numpy())
                all_train_outputs.append(output_batch.cpu().numpy())

            all_train_predictions = np.concatenate(all_train_predictions)
            all_train_outputs = np.concatenate(all_train_outputs)

            train_mse = mean_squared_error(all_train_outputs, all_train_predictions)
            train_r2 = r2_score(all_train_outputs, all_train_predictions)

            train_mse_scores.append(train_mse)
            train_r2_scores.append(train_r2)

            all_val_predictions = []
            all_val_outputs = []
            total_valid_loss = 0.0

            for branch_batch, trunk_batch, output_batch in valid_loader:
                branch_batch = branch_batch.to(device)
                trunk_batch = trunk_batch.to(device)
                output_batch = output_batch.to(device)

                qt_predictions = deeponet_model(branch_batch, trunk_batch)
                all_val_predictions.append(qt_predictions.cpu().numpy())
                all_val_outputs.append(output_batch.cpu().numpy())

                loss = criterion(qt_predictions, output_batch)
                total_valid_loss += loss.item()

            all_val_predictions = np.concatenate(all_val_predictions)
            all_val_outputs = np.concatenate(all_val_outputs)

            val_mse = mean_squared_error(all_val_outputs, all_val_predictions)
            val_r2 = r2_score(all_val_outputs, all_val_predictions)
            valid_losses.append(total_valid_loss / len(valid_loader))

        print(f'第 [{epoch + 1}/{epochs}] 轮，训练损失: {total_loss_epoch / len(train_loader):.4f}, '
              f'训练 MSE: {train_mse:.4f}, 训练 R²: {train_r2:.4f}, '
              f'验证 MSE: {val_mse:.4f}, 验证 R²: {val_r2:.4f}')

torch.save(deeponet_model.state_dict(), model_save_path)

save_dir = './data8'
os.makedirs(save_dir, exist_ok=True)

def evaluate_model(model, dataset, batch_size, dataset_name):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []
    all_true_outputs = []

    with torch.no_grad():
        for branch_inputs, trunk_inputs, true_outputs in data_loader:
            branch_inputs = branch_inputs.to(device)
            trunk_inputs = trunk_inputs.to(device)
            true_outputs = true_outputs.to(device)

            qt_predictions = model(branch_inputs, trunk_inputs)

            all_predictions.append(qt_predictions.cpu().numpy())
            all_true_outputs.append(true_outputs.cpu().numpy())

    all_predictions = np.concatenate(all_predictions)
    all_true_outputs = np.concatenate(all_true_outputs)

    mse = mean_squared_error(all_true_outputs, all_predictions)
    mae = mean_absolute_error(all_true_outputs, all_predictions)
    r2 = r2_score(all_true_outputs, all_predictions)

    print(f"{dataset_name} MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    return mse, mae, r2, all_predictions, all_true_outputs

def save_evaluation_results(model, dataset, batch_size, output_path, csv_path, dataset_name):
    mse, mae, r2, predictions, true_outputs = evaluate_model(model, dataset, batch_size, dataset_name)
    torch.save(predictions, output_path)

    true_values = true_outputs.squeeze()
    predictions = predictions.squeeze()
    results_df = pd.DataFrame({'True Values': true_values, 'Predicted Values': predictions})
    results_df.to_csv(csv_path, index=False)

    print(f"{dataset_name} MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    print(f"{dataset_name} 结果已保存到 {csv_path}")

save_evaluation_results(deeponet_model, train_dataset, batch_size, train_pred_save_path, 'train_results.csv', '训练集')
save_evaluation_results(deeponet_model, valid_dataset, batch_size, val_pred_save_path, 'valid_results.csv', '验证集')
save_evaluation_results(deeponet_model, test_dataset, batch_size, test_pred_save_path, 'test_results.csv', '测试集')

def plot_losses(total_losses, filename):
    epochs_range = range(1, len(total_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, total_losses, label='Total Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'data8/{filename}.png')
    plt.show()

plot_losses(total_losses, 'loss_plot')

def plot_metrics(epochs, mse_scores, r2_scores, filename):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1, 10), mse_scores, label='Training MSE', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Training MSE Over Epochs')
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1, 10), r2_scores, label='Training R²', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('R² Score')
    plt.title('Training R² Over Epochs')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'data8/{filename}.png')
    plt.show()

plot_metrics(epochs, train_mse_scores, train_r2_scores, 'training_metrics')

def plot_predictions(true_values, predicted_values, title, filename):
    plt.figure(figsize=(8, 6))
    plt.scatter(true_values, predicted_values, color='blue', alpha=0.6)
    plt.plot(
        [min(true_values), max(true_values)],
        [min(true_values), max(true_values)],
        color='red',
        linestyle='--',
        linewidth=2,
    )
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.grid(True)
    save_dir = "data8"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{filename}.png")
    plt.savefig(save_path)
    plt.show()
    plt.close()

def load_and_save_plots():
    if os.path.exists(train_pred_save_path):
        train_true = outputs_tensor.cpu().numpy().squeeze()
        train_pred = torch.load(train_pred_save_path)
        plot_predictions(train_true, train_pred, "Training Set: Predicted vs True", "train_predictions")
    if os.path.exists(val_pred_save_path):
        val_true = val_outputs_tensor.cpu().numpy().squeeze()
        val_pred = torch.load(val_pred_save_path)
        plot_predictions(val_true, val_pred, "Validation Set: Predicted vs True", "val_predictions")
    if os.path.exists(test_pred_save_path):
        test_true = test_outputs_tensor.cpu().numpy().squeeze()
        test_pred = torch.load(test_pred_save_path)
        plot_predictions(test_true, test_pred, "Test Set: Predicted vs True", "test_predictions")

try:
    load_and_save_plots()
    print("success")
except Exception as e:
    print(f"error: {e}")
