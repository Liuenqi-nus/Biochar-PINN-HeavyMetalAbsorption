# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# import pandas as pd
# import numpy as np
# import pickle
# import json
# import os
#
# # 设置设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
#
# # 定义分支和主干网络结构，与训练时一致
# class BranchNetwork(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(BranchNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
#         self.fc3 = nn.Linear(hidden_size // 2, hidden_size //4)
#         self.fc4 = nn.Linear(hidden_size //4, hidden_size // 8)
#         self.fc5 = nn.Linear(hidden_size //8, output_size)
#         self.relu = nn.Tanh()
#
#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.relu(self.fc3(x))
#         x = self.relu(self.fc4(x))
#         x = self.fc5(x)
#         return x
#
# class TrunkNetwork(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(TrunkNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
#         self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
#         self.fc4 = nn.Linear(hidden_size // 4, hidden_size // 8)
#         self.fc5 = nn.Linear(hidden_size // 8, output_size)
#         self.relu = nn.Tanh()
#
#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.relu(self.fc3(x))
#         x = self.relu(self.fc4(x))
#         x = self.fc5(x)
#         return x
#
# class DeepONet(nn.Module):
#     def __init__(self, branch_input_size, trunk_input_size, hidden_size, output_size):
#         super(DeepONet, self).__init__()
#         self.output_size = output_size
#         self.hidden_size = hidden_size
#
#         self.branch_net = BranchNetwork(branch_input_size, hidden_size, output_size * hidden_size)
#         self.trunk_net = TrunkNetwork(trunk_input_size, hidden_size, hidden_size)
#
#         # 在预测时，我们不需要可学习的 lambda 参数
#         # 如果您在模型中定义了这些参数，需要在这里也定义，但不影响预测
#         self.lambda_data = nn.Parameter(torch.tensor(1.0))
#         self.lambda_physics = nn.Parameter(torch.tensor(100.0))
#         self.lambda_boundary = nn.Parameter(torch.tensor(100.0))
#         self.lambda_monotonicity = nn.Parameter(torch.tensor(10.0))
#
#     def forward(self, branch_input, trunk_input, C0_values):
#         batch_size = branch_input.size(0)
#
#         branch_output = self.branch_net(branch_input)  # (batch_size, output_size * hidden_size)
#         trunk_output = self.trunk_net(trunk_input)     # (batch_size, hidden_size)
#
#         branch_output = branch_output.view(batch_size, self.output_size, self.hidden_size)
#         trunk_output = trunk_output.view(batch_size, self.hidden_size, 1)
#
#         combined_output = torch.bmm(branch_output, trunk_output).squeeze(2)
#
#         # 提取各个输出
#         delta_C = combined_output[:, 0:1]
#         qe = combined_output[:, 1:2]
#         k2 = combined_output[:, 2:3]
#         Ct = combined_output[:, 3:4]
#
#         return delta_C, qe, k2, Ct
#
# # 加载训练好的模型参数
# model_save_path = 'Data7/deeponet_pinn_model_epoch_8000.pth'  # 请根据实际路径调整
# branch_input_size = 27  # 请根据您的模型设置调整
# trunk_input_size = 1    # 请根据您的模型设置调整
# hidden_size = 64
# output_size = 4
#
# # 实例化模型并加载参数
# deeponet_model = DeepONet(branch_input_size, trunk_input_size, hidden_size, output_size).to(device)
# deeponet_model.load_state_dict(torch.load(model_save_path, map_location=device))
# deeponet_model.eval()  # 设置模型为评估模式
#
# # 加载预处理器（标准化器、标签编码器等）
# with open('data8/branch_scaler.pkl', 'rb') as f:
#     branch_scaler = pickle.load(f)
#
# with open('data8/trunk_scaler.pkl', 'rb') as f:
#     trunk_scaler = pickle.load(f)
#
# with open('label_encoders.pkl', 'rb') as f:
#     label_encoders = pickle.load(f)
#
# # 加载金属离子特征
# with open('metal_ion_features.json', 'r') as f:
#     metal_ion_features = json.load(f)
#
# # 定义用于编码文本特征的函数
# def encode_text_features(data, label_encoders, is_train=False):
#     # 提取文本特征（包括 metal ions）
#     text_features = data[['atmosphere', 'biomass type', 'methods', 'metal ions']]
#     encoded_text_features = []
#
#     # 对非金属元素的文本特征进行编码
#     for col in text_features.columns[:-1]:  # 不对 'metal ions' 列进行编码
#         le = label_encoders[col]
#         encoded_col = le.transform(text_features[col])
#         encoded_text_features.append(encoded_col)
#
#     # 处理 'metal ions' 列并生成金属特征
#     metal_feature_data = []
#     for idx, metal in text_features['metal ions'].items():
#         if metal in metal_ion_features:
#             metal_content = 1  # 金属含量设置为1，因为每行仅有一个金属类型
#             metal_features = metal_ion_features[metal]  # 获取金属特征
#             metal_feature_data.append([metal_content] + metal_features)
#         else:
#             metal_feature_data.append([0] * 7)  # 如果没有匹配到金属，填充0值
#
#     # 将所有特征合并为一个数组
#     encoded_text_features = np.column_stack(encoded_text_features)
#     metal_feature_data = np.array(metal_feature_data)
#     encoded_text_features = np.hstack((encoded_text_features, metal_feature_data))
#
#     # 使用 pd.concat 一次性添加金属特征到原始 DataFrame 中
#     feature_names = ['content', 'radius', 'electronegativity', 'max_oxidation_state', 'hydration_energy', 'group', 'period']
#     metal_feature_df = pd.DataFrame(metal_feature_data, columns=[f'metal_{feature_name}' for feature_name in feature_names], index=data.index)
#     data = pd.concat([data, metal_feature_df], axis=1)
#
#     return encoded_text_features, data
#
# # 定义分支和主干输入的列名
# branch_columns = ['T1', 'T2', 'T3', 'pH', 'time1', 'time2', 'Fe3', 'Fe2', 'Ag', 'Al3', 'Ce3', 'Cu2', 'La3', 'Mn2',
#                   'Mn7', 'Mg2', 'Zn2', 'BC', 'C0', 'mg']  # 原始的非金属特征列
# branch_columns.extend([f'metal_{feature_name}' for feature_name in ['content', 'radius', 'electronegativity', 'max_oxidation_state', 'hydration_energy', 'group', 'period']])
# trunk_columns = ['time3']
#
# # 假设您有新的数据需要进行预测
# # 请将 new_data.csv 替换为您的新数据文件路径
# new_data_file = 'predict2/Ag.csv'  # 新的数据文件路径
# new_data = pd.read_csv(new_data_file)
#
# # 预处理新的数据
# # 对文本特征进行编码，并更新 DataFrame
# new_encoded_text_features, new_data = encode_text_features(new_data, label_encoders, is_train=False)
#
# # 提取分支和主干输入
# branch_inputs = new_data[branch_columns].values
# trunk_inputs = new_data[trunk_columns].values
# C0_values = new_data[['C0']].values  # 提取初始浓度 C0
#
# # 标准化输入
# branch_inputs_scaled = branch_scaler.transform(branch_inputs)
# trunk_inputs_scaled = trunk_scaler.transform(trunk_inputs)
#
# # 将标准化后的输入转换为 PyTorch 张量并移动到设备上
# branch_inputs_tensor = torch.tensor(branch_inputs_scaled, dtype=torch.float32).to(device)
# trunk_inputs_tensor = torch.tensor(trunk_inputs_scaled, dtype=torch.float32).to(device)
# C0_tensor = torch.tensor(C0_values, dtype=torch.float32).to(device)
#
# # 预测
# with torch.no_grad():
#     # 前向传播
#     delta_C_predictions, _, _, Ct_predictions = deeponet_model(branch_inputs_tensor, trunk_inputs_tensor, C0_tensor)
#
#     # 提取 mg 列并反标准化
#     mg_index = branch_columns.index('mg')
#     mg_batch = branch_inputs_tensor[:, mg_index:mg_index + 1]
#
#     mg_mean = torch.tensor(branch_scaler.mean_[mg_index], device=device)
#     mg_std = torch.tensor(branch_scaler.scale_[mg_index], device=device)
#     mg_batch_original = mg_batch * mg_std + mg_mean
#
#     # 计算 qt 预测值
#     qt_predictions = delta_C_predictions / mg_batch_original
#
#     # 将预测值从 GPU 移动到 CPU，并转换为 numpy 数组
#     qt_predictions_np = qt_predictions.cpu().numpy()
#     Ct_predictions_np = Ct_predictions.cpu().numpy()
#
# # 将预测结果添加到 DataFrame 中
# new_data['Predicted_qt'] = qt_predictions_np
# new_data['Predicted_Ct'] = Ct_predictions_np
#
# # 保存预测结果到 CSV 文件
# output_csv_path = 'prediction_results.csv'
# new_data.to_csv(output_csv_path, index=False)
# print(f"预测结果已保存到 {output_csv_path}")
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import json
import pickle
import os
from datetime import datetime
import argparse

# =========================
# Define Model Architectures
# =========================

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
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.branch_net = BranchNetwork(branch_input_size, hidden_size, output_size * hidden_size)
        self.trunk_net = TrunkNetwork(trunk_input_size, hidden_size, hidden_size)

        self.lambda_data = nn.Parameter(torch.tensor(1.0))
        self.lambda_physics = nn.Parameter(torch.tensor(100.0))
        self.lambda_boundary = nn.Parameter(torch.tensor(100.0))
        self.lambda_monotonicity = nn.Parameter(torch.tensor(10.0))
    def forward(self, branch_input, trunk_input, C0_values):
        batch_size = branch_input.size(0)

        # Branch and trunk outputs
        branch_output = self.branch_net(branch_input)  # (batch_size, output_size * hidden_size)
        trunk_output = self.trunk_net(trunk_input)     # (batch_size, hidden_size)

        # Reshape for batch matrix multiplication
        branch_output = branch_output.view(batch_size, self.output_size, self.hidden_size)  # (batch_size, output_size, hidden_size)
        trunk_output = trunk_output.view(batch_size, self.hidden_size, 1)                   # (batch_size, hidden_size, 1)

        # Batch matrix multiplication
        combined_output = torch.bmm(branch_output, trunk_output).squeeze(2)                 # (batch_size, output_size)

        # Extract outputs
        delta_C = combined_output[:, 0:1]
        qe = F.softplus(combined_output[:, 1:2])
        k2 = F.softplus(combined_output[:, 2:3])
        Ct = F.softplus(combined_output[:, 3:4])

        return delta_C, qe, k2, Ct

# =========================
# Preprocessing Function
# =========================

def encode_text_features(data, label_encoders, is_train=True, metal_ion_features=None):
    # Extract text features
    text_features = data[['atmosphere', 'biomass type', 'methods', 'metal ions']]
    encoded_text_features = []
    encoded_text_feature_names = []

    # Encode non-metal text features
    for col in text_features.columns[:-1]:  # Exclude 'metal ions'
        le = label_encoders[col]
        encoded_col = le.transform(text_features[col])
        encoded_text_features.append(encoded_col)
        encoded_text_feature_names.append(f"{col}_encoded")

    # Process 'metal ions' and generate metal features
    metal_feature_data = []
    for metal in text_features['metal ions']:
        if metal in metal_ion_features:
            metal_content = 1  # Assuming each row has only one metal type
            metal_features = metal_ion_features[metal]
            metal_feature_data.append([metal_content] + metal_features)
        else:
            metal_feature_data.append([0] * 7)  # Assuming 7 metal features

    metal_feature_names = [f'metal_{feature}' for feature in ['content', 'radius', 'electronegativity',
                                                              'max_oxidation_state', 'hydration_energy', 'group', 'period']]
    metal_feature_data = np.array(metal_feature_data)
    metal_feature_df = pd.DataFrame(metal_feature_data, columns=metal_feature_names, index=data.index)

    # Encode text features
    encoded_text_features = np.column_stack(encoded_text_features)
    encoded_text_features_df = pd.DataFrame(encoded_text_features, columns=encoded_text_feature_names, index=data.index)

    # Merge all features
    data = pd.concat([data, encoded_text_features_df, metal_feature_df], axis=1)

    return encoded_text_features, data, encoded_text_feature_names, metal_feature_names

# =========================
# Prediction Function
# =========================

def predict(model, branch_scaler, trunk_scaler, label_encoders, metal_ion_features,
            input_data_path, output_csv_path, output_qt_path, output_Ct_path, device):
    # Load new data
    new_data = pd.read_csv(input_data_path)
    original_data = new_data.copy()  # Keep a copy for reference

    # Encode text features
    _, processed_data, encoded_text_feature_names, metal_feature_names = encode_text_features(
        new_data, label_encoders, is_train=False, metal_ion_features=metal_ion_features
    )
    # 打印处理后的DataFrame列名
    print("Processed data columns:", processed_data.columns)

    # Define branch and trunk columns
    branch_columns = ['T1', 'T2', 'T3', 'pH', 'time1', 'time2', 'Fe3', 'Fe2', 'Ag', 'Al3', 'Ce3', 'Cu2', 'La3', 'Mn2',
                      'Mn7', 'Mg2', 'Zn2', 'BC', 'C0', 'mg']
    branch_columns.extend(metal_feature_names)
    branch_columns.extend(encoded_text_feature_names)

    trunk_columns = ['time3']

    # Ensure all necessary columns are present
    missing_columns = set(branch_columns + trunk_columns) - set(processed_data.columns)
    if missing_columns:
        raise ValueError(f"The following required columns are missing from the input data: {missing_columns}")

    # 在定义完成后直接打印使用的特征列：
    print("Using these branch features for prediction:", branch_columns)
    print("Using these trunk features for prediction:", trunk_columns)

    # 提取分支和主干输入
    branch_inputs = processed_data[branch_columns].values
    trunk_inputs = processed_data[trunk_columns].values

    # 查看提取出来的输入维度和第一行数据，以核对特征是否正确
    print("Branch inputs shape:", branch_inputs.shape)
    print("Trunk inputs shape:", trunk_inputs.shape)
    print("A sample of branch inputs (first row):", branch_inputs[0])
    print("A sample of trunk inputs (first row):", trunk_inputs[0])

    # Scale inputs
    branch_inputs_scaled = branch_scaler.transform(branch_inputs)
    trunk_inputs_scaled = trunk_scaler.transform(trunk_inputs)

    # Convert to tensors
    branch_inputs_tensor = torch.tensor(branch_inputs_scaled, dtype=torch.float32).to(device)
    trunk_inputs_tensor = torch.tensor(trunk_inputs_scaled, dtype=torch.float32).to(device)

    # Extract C0 values
    C0_values = processed_data[['C0']].values
    C0_tensor = torch.tensor(C0_values, dtype=torch.float32).to(device)

    # Perform prediction
    with torch.no_grad():
        delta_C, qe, k2, Ct = model(branch_inputs_tensor, trunk_inputs_tensor, C0_tensor)
        # Compute qt
        mg_index = branch_columns.index('mg')
        mg_batch = branch_inputs_tensor[:, mg_index:mg_index+1]
        mg_mean = branch_scaler.mean_[mg_index]
        mg_std = branch_scaler.scale_[mg_index]
        mg_original = mg_batch * mg_std + mg_mean
        qt = delta_C / mg_original

    # Move predictions to CPU and convert to numpy
    qt = qt.cpu().numpy()
    Ct = Ct.cpu().numpy()

    # Add predictions to the original DataFrame
    original_data['Predicted_qt'] = qt
    original_data['Predicted_Ct'] = Ct

    # Save predictions to CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    original_data.to_csv(output_csv_path, index=False)

    # Optionally save predictions as tensors
    torch.save(qt, output_qt_path)
    torch.save(Ct, output_Ct_path)

    print(f"Predictions saved to {output_csv_path}")
    print(f"qt predictions saved to {output_qt_path}")
    print(f"Ct predictions saved to {output_Ct_path}")

# =========================
# Main Function
# =========================

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load label encoders
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)

    # Load scalers
    with open('data8/branch_scaler.pkl', 'rb') as f:
        branch_scaler = pickle.load(f)
    with open('data8/trunk_scaler.pkl', 'rb') as f:
        trunk_scaler = pickle.load(f)

    # Load metal ion features
    with open('metal_ion_features.json', 'r') as f:
        metal_ion_features = json.load(f)

    # Define model parameters (ensure these match training)
    # These should match the training script's parameters
    branch_input_size = branch_scaler.mean_.shape[0]
    trunk_input_size = trunk_scaler.mean_.shape[0]
    hidden_size = 64
    output_size = 4  # delta_C, qe, k2, Ct

    # Initialize model
    model = DeepONet(branch_input_size, trunk_input_size, hidden_size, output_size).to(device)

    # Load model weights
    model.load_state_dict(torch.load('Data7/deeponet_pinn_model_epoch_4000.pth', map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # Perform prediction
    predict(
        model=model,
        branch_scaler=branch_scaler,
        trunk_scaler=trunk_scaler,
        label_encoders=label_encoders,
        metal_ion_features=metal_ion_features,
        input_data_path=args.input,
        output_csv_path=args.output_csv,
        output_qt_path=args.output_qt,
        output_Ct_path=args.output_Ct,
        device=device
    )

# =========================
# Entry Point
# =========================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DeepONet Prediction Script")
    parser.add_argument('--input', type=str, default='top_100_solutions_Sb1.csv',
                        help='Path to the input CSV file for prediction.')
    parser.add_argument('--output_csv', type=str, default='data8/predictions.csv',
                        help='Path to save the output predictions CSV.')
    parser.add_argument('--output_qt', type=str, default='data8/predictions_qt.pt',
                        help='Path to save the qt predictions tensor.')
    parser.add_argument('--output_Ct', type=str, default='data8/predictions_Ct.pt',
                        help='Path to save the Ct predictions tensor.')
    args = parser.parse_args()
    main(args)
