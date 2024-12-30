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

# 设置随机种子
torch.manual_seed(11)
np.random.seed(11)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(11)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 获取当前时间戳
def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# 从 JSON 文件加载金属特征
with open('metal_ion_features.json', 'r') as f:
    metal_ion_features = json.load(f)

# 保存模型和预测的路径
model_save_path = 'data8/deeponet_model.pth'
train_pred_save_path_qt = 'data8/train_predictions_qt.pt'
train_pred_save_path_Ct = 'data8/train_predictions_Ct.pt'
val_pred_save_path_qt = 'data8/val_predictions_qt.pt'
val_pred_save_path_Ct = 'data8/val_predictions_Ct.pt'
test_pred_save_path_qt = 'data8/test_predictions_qt.pt'
test_pred_save_path_Ct = 'data8/test_predictions_Ct.pt'

# 加载数据
train_file_path = 'data9/train.csv'
valid_file_path = 'data9/test1.csv'
test_file_path = 'data9/valid.csv'

train_data = pd.read_csv(train_file_path)
valid_data = pd.read_csv(valid_file_path)
test_data = pd.read_csv(test_file_path)

def encode_text_features(data, label_encoders=None, is_train=True):
    # 提取文本特征（包括 'metal ions'）
    text_features = data[['atmosphere', 'biomass type', 'methods', 'metal ions']]
    encoded_text_features = []
    encoded_text_feature_names = []

    # 对非金属的文本特征进行编码
    if is_train:
        label_encoders = {}
        for col in text_features.columns[:-1]:  # 不包括 'metal ions'
            le = LabelEncoder()
            encoded_col = le.fit_transform(text_features[col])
            label_encoders[col] = le
            encoded_text_features.append(encoded_col)
            encoded_text_feature_names.append(f"{col}_encoded")
    else:
        for col in text_features.columns[:-1]:
            encoded_col = label_encoders[col].transform(text_features[col])
            encoded_text_features.append(encoded_col)
            encoded_text_feature_names.append(f"{col}_encoded")

    # 处理 'metal ions' 列并生成金属特征
    metal_feature_data = []
    for idx, metal in text_features['metal ions'].items():
        if metal in metal_ion_features:
            metal_content = 1  # 金属含量设置为1
            metal_features = metal_ion_features[metal]  # 获取金属特征
            metal_feature_data.append([metal_content] + metal_features)
        else:
            metal_feature_data.append([0] * 7)  # 如果没有匹配到金属，填充0值

    # 将编码后的文本特征转换为 DataFrame
    encoded_text_features = np.column_stack(encoded_text_features)
    encoded_text_features_df = pd.DataFrame(encoded_text_features, columns=encoded_text_feature_names, index=data.index)

    # 将金属特征转换为 DataFrame
    metal_feature_names = [f'metal_{feature_name}' for feature_name in [
        'content', 'radius', 'electronegativity', 'max_oxidation_state',
        'hydration_energy', 'group', 'period'
    ]]
    metal_feature_data = np.array(metal_feature_data)
    metal_feature_df = pd.DataFrame(metal_feature_data, columns=metal_feature_names, index=data.index)

    # 将所有新特征合并到原始 DataFrame 中
    data = pd.concat([data, encoded_text_features_df, metal_feature_df], axis=1)

    return encoded_text_features, label_encoders, data, encoded_text_feature_names, metal_feature_names

# 对训练、验证和测试集进行处理，并更新 DataFrame
train_encoded_text_features, label_encoders, train_data, encoded_text_feature_names, metal_feature_names = encode_text_features(train_data, is_train=True)
val_encoded_text_features, _, valid_data, _, _ = encode_text_features(valid_data, label_encoders, is_train=False)
test_encoded_text_features, _, test_data, _, _ = encode_text_features(test_data, label_encoders, is_train=False)

# 动态构建特征列名
branch_columns = [
    ' T1', 'T2', 'T3', 'pH', 'time1', 'time2',
    'Fe3', 'Fe2', 'Ag', 'Al3', 'Ce3', 'Cu2',
    'La3', 'Mn2', 'Mn7', 'Mg2', 'Zn2', 'BC', 'C0', 'mg'
]
branch_columns.extend(metal_feature_names)        # 添加金属特征列名
branch_columns.extend(encoded_text_feature_names) # 添加编码文本特征列名

print("Branch columns:", branch_columns)

# 保存标签编码器
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

# 定义分支和主干输入，以及输出
trunk_columns = ['time3']

# 训练数据的分支和主干输入
branch_inputs = train_data[branch_columns].values
trunk_inputs = train_data[trunk_columns].values
outputs_qt = train_data[['qt']].values
outputs_Ct = train_data[['Ct']].values
C0_train = train_data[['C0']].values  # 提取初始浓度 C0

# 验证数据的分支和主干输入
valid_branch_inputs = valid_data[branch_columns].values
valid_trunk_inputs = valid_data[trunk_columns].values
valid_outputs_qt = valid_data[['qt']].values
valid_outputs_Ct = valid_data[['Ct']].values
C0_valid = valid_data[['C0']].values

# 测试数据的分支和主干输入
test_branch_inputs = test_data[branch_columns].values
test_trunk_inputs = test_data[trunk_columns].values
test_outputs_qt = test_data[['qt']].values
test_outputs_Ct = test_data[['Ct']].values
C0_test = test_data[['C0']].values

# ---------------------------
# 根据金属特征分组并设置不同放大倍数
# ---------------------------
# 假设 Group A 和 Group B, 并为它们定义不同的放大因子
group_a = ['Fe3', 'Fe2']  # 举例：Group A
group_b = ['Ag', 'Al3', 'Ce3', 'Cu2', 'La3', 'Mn2', 'Mn7', 'Mg2', 'Zn2']  # Group B

factor_a = 1.0
factor_b = 1.0

# 获取非金属列
non_metals_cols = [col for col in branch_columns if col not in (group_a + group_b)]
non_metal_indices = [branch_columns.index(col) for col in non_metals_cols]

# 获取分组 A、B 的列索引
group_a_indices = [branch_columns.index(col) for col in group_a]
group_b_indices = [branch_columns.index(col) for col in group_b]

# 拆分并只对非金属做标准化，对金属各自放大
branch_non_metal_train = branch_inputs[:, non_metal_indices]
branch_metal_a_train = branch_inputs[:, group_a_indices]
branch_metal_b_train = branch_inputs[:, group_b_indices]

branch_scaler_non_metal = StandardScaler()
branch_non_metal_train_scaled = branch_scaler_non_metal.fit_transform(branch_non_metal_train)

branch_metal_a_train_scaled = branch_metal_a_train * factor_a
branch_metal_b_train_scaled = branch_metal_b_train * factor_b

# 重新合并到原来的维度顺序
branch_inputs_train_scaled = np.zeros_like(branch_inputs)
branch_inputs_train_scaled[:, non_metal_indices] = branch_non_metal_train_scaled
branch_inputs_train_scaled[:, group_a_indices] = branch_metal_a_train_scaled
branch_inputs_train_scaled[:, group_b_indices] = branch_metal_b_train_scaled

# 验证集
branch_non_metal_val = valid_branch_inputs[:, non_metal_indices]
branch_metal_a_val = valid_branch_inputs[:, group_a_indices]
branch_metal_b_val = valid_branch_inputs[:, group_b_indices]

branch_non_metal_val_scaled = branch_scaler_non_metal.transform(branch_non_metal_val)
branch_metal_a_val_scaled = branch_metal_a_val * factor_a
branch_metal_b_val_scaled = branch_metal_b_val * factor_b

branch_inputs_val_scaled = np.zeros_like(valid_branch_inputs)
branch_inputs_val_scaled[:, non_metal_indices] = branch_non_metal_val_scaled
branch_inputs_val_scaled[:, group_a_indices] = branch_metal_a_val_scaled
branch_inputs_val_scaled[:, group_b_indices] = branch_metal_b_val_scaled

# 测试集
branch_non_metal_test = test_branch_inputs[:, non_metal_indices]
branch_metal_a_test = test_branch_inputs[:, group_a_indices]
branch_metal_b_test = test_branch_inputs[:, group_b_indices]

branch_non_metal_test_scaled = branch_scaler_non_metal.transform(branch_non_metal_test)
branch_metal_a_test_scaled = branch_metal_a_test * factor_a
branch_metal_b_test_scaled = branch_metal_b_test * factor_b

branch_inputs_test_scaled = np.zeros_like(test_branch_inputs)
branch_inputs_test_scaled[:, non_metal_indices] = branch_non_metal_test_scaled
branch_inputs_test_scaled[:, group_a_indices] = branch_metal_a_test_scaled
branch_inputs_test_scaled[:, group_b_indices] = branch_metal_b_test_scaled

# 对 trunk_inputs 统一使用 StandardScaler
trunk_scaler = StandardScaler()
trunk_inputs_scaled = trunk_scaler.fit_transform(trunk_inputs)
valid_trunk_inputs_scaled = trunk_scaler.transform(valid_trunk_inputs)
test_trunk_inputs_scaled = trunk_scaler.transform(test_trunk_inputs)

# 保存标准化器
with open('data8/branch_scaler_non_metal.pkl', 'wb') as f:
    pickle.dump(branch_scaler_non_metal, f)

with open('data8/trunk_scaler.pkl', 'wb') as f:
    pickle.dump(trunk_scaler, f)

print("Train DataFrame columns:", train_data.columns.tolist())
print("Branch columns:", branch_columns)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 转换为tensor
branch_inputs_tensor = torch.tensor(branch_inputs_train_scaled, dtype=torch.float32).to(device)
trunk_inputs_tensor = torch.tensor(trunk_inputs_scaled, dtype=torch.float32).to(device)
outputs_qt_tensor = torch.tensor(outputs_qt, dtype=torch.float32).to(device)
outputs_Ct_tensor = torch.tensor(outputs_Ct, dtype=torch.float32).to(device)
C0_train_tensor = torch.tensor(C0_train, dtype=torch.float32).to(device)

val_branch_inputs_tensor = torch.tensor(branch_inputs_val_scaled, dtype=torch.float32).to(device)
val_trunk_inputs_tensor = torch.tensor(valid_trunk_inputs_scaled, dtype=torch.float32).to(device)
val_outputs_qt_tensor = torch.tensor(valid_outputs_qt, dtype=torch.float32).to(device)
val_outputs_Ct_tensor = torch.tensor(valid_outputs_Ct, dtype=torch.float32).to(device)
C0_valid_tensor = torch.tensor(C0_valid, dtype=torch.float32).to(device)

test_branch_inputs_tensor = torch.tensor(branch_inputs_test_scaled, dtype=torch.float32).to(device)
test_trunk_inputs_tensor = torch.tensor(test_trunk_inputs_scaled, dtype=torch.float32).to(device)
test_outputs_qt_tensor = torch.tensor(test_outputs_qt, dtype=torch.float32).to(device)
test_outputs_Ct_tensor = torch.tensor(test_outputs_Ct, dtype=torch.float32).to(device)
C0_test_tensor = torch.tensor(C0_test, dtype=torch.float32).to(device)

# 创建 DataLoader
batch_size = 32
train_dataset = TensorDataset(branch_inputs_tensor, trunk_inputs_tensor,
                              outputs_qt_tensor, outputs_Ct_tensor, C0_train_tensor)
valid_dataset = TensorDataset(val_branch_inputs_tensor, val_trunk_inputs_tensor,
                              val_outputs_qt_tensor, val_outputs_Ct_tensor, C0_valid_tensor)
test_dataset = TensorDataset(test_branch_inputs_tensor, test_trunk_inputs_tensor,
                             test_outputs_qt_tensor, test_outputs_Ct_tensor, C0_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class BranchNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BranchNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.fc5 = nn.Linear(hidden_size // 2, output_size)
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
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.fc5 = nn.Linear(hidden_size // 2, output_size)
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

        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)

        branch_output = branch_output.view(batch_size, self.output_size, self.hidden_size)
        trunk_output = trunk_output.view(batch_size, self.hidden_size, 1)

        combined_output = torch.bmm(branch_output, trunk_output).squeeze(2)

        delta_C = combined_output[:, 0:1]
        qe = F.softplus(combined_output[:, 1:2])
        k2 = F.softplus(combined_output[:, 2:3])
        Ct = F.softplus(combined_output[:, 3:4])

        return delta_C, qe, k2, Ct

criterion = nn.MSELoss()
deeponet_model = DeepONet(branch_inputs.shape[1], trunk_inputs.shape[1], 128, 4).to(device)
optimizer = torch.optim.Adam(deeponet_model.parameters(), lr=0.0005)

# 找到 mg_index
mg_index = branch_columns.index('mg')

# physics_loss、monotonicity_loss、boundary_loss 中若需要对 mg 特征做反标准化
# 则要根据其是否在 group_a 或 group_b 中来决定乘 factor_a / factor_b 或者从 StandardScaler 中读均值方差。
# 以下示例仅展示如何获取 mg_batch_original，根据你的实际业务做适配。
def physics_loss_pseudo_second_order(trunk_input_scaled, delta_C_predictions, qe_predictions, k2_predictions,
                                     time_scaler, mg_batch, C0_values, Ct_predictions, mg_index, branch_scaler):
    delta_C_grad_scaled = torch.autograd.grad(
        delta_C_predictions,
        trunk_input_scaled,
        grad_outputs=torch.ones_like(delta_C_predictions),
        create_graph=True,
        retain_graph=True,
    )[0]

    std_t = torch.tensor(time_scaler.scale_[0], dtype=delta_C_grad_scaled.dtype, device=delta_C_grad_scaled.device)
    delta_C_grad = delta_C_grad_scaled / std_t

    # 判断 mg 属于 group A、group B 还是非金属
    if branch_columns[mg_index] in group_a:
        mg_batch_original = mg_batch * factor_a
    elif branch_columns[mg_index] in group_b:
        mg_batch_original = mg_batch * factor_b
    else:
        # mg 不在任何分组时(非金属), 使用非金属的标准化参数
        mg_mean = branch_scaler_non_metal.mean_[non_metals_cols.index('mg')]
        mg_std = branch_scaler_non_metal.scale_[non_metals_cols.index('mg')]
        mg_batch_original = mg_batch * mg_std + mg_mean

    qt_grad = delta_C_grad / mg_batch_original
    qt_predictions = delta_C_predictions / mg_batch_original

    physical_constraint = qt_grad - k2_predictions * (qe_predictions - qt_predictions) ** 2
    physics_informed_loss = torch.mean(physical_constraint ** 2)

    physical_constraint_Ct = delta_C_predictions - (C0_values - Ct_predictions)
    physics_informed_loss_Ct = torch.mean(physical_constraint_Ct ** 2)

    total_physics_loss = physics_informed_loss + physics_informed_loss_Ct
    return total_physics_loss

def monotonicity_loss(delta_C_predictions, trunk_input_scaled, time_scaler, mg_batch, mg_index, branch_scaler):
    delta_C_grad_scaled = torch.autograd.grad(
        delta_C_predictions,
        trunk_input_scaled,
        grad_outputs=torch.ones_like(delta_C_predictions),
        create_graph=True,
        retain_graph=True,
    )[0]

    std_t = torch.tensor(time_scaler.scale_[0], dtype=delta_C_grad_scaled.dtype, device=delta_C_grad_scaled.device)
    delta_C_grad = delta_C_grad_scaled / std_t

    if branch_columns[mg_index] in group_a:
        mg_batch_original = mg_batch * factor_a
    elif branch_columns[mg_index] in group_b:
        mg_batch_original = mg_batch * factor_b
    else:
        mg_mean = branch_scaler_non_metal.mean_[non_metals_cols.index('mg')]
        mg_std = branch_scaler_non_metal.scale_[non_metals_cols.index('mg')]
        mg_batch_original = mg_batch * mg_std + mg_mean

    qt_grad = delta_C_grad / mg_batch_original
    violation = F.relu(-qt_grad)
    monotonicity_loss_value = torch.mean(violation ** 2)
    return monotonicity_loss_value

def boundary_loss(deeponet_model, branch_input, C0_values, mg_index, branch_scaler):
    t_zero = torch.zeros(branch_input.size(0), 1, device=branch_input.device, requires_grad=True)
    delta_C_zero, _, _, Ct_zero = deeponet_model(branch_input, t_zero, C0_values)

    mg_batch = branch_input[:, mg_index:mg_index + 1]
    if branch_columns[mg_index] in group_a:
        mg_batch_original = mg_batch * factor_a
    elif branch_columns[mg_index] in group_b:
        mg_batch_original = mg_batch * factor_b
    else:
        mg_mean = branch_scaler_non_metal.mean_[non_metals_cols.index('mg')]
        mg_std = branch_scaler_non_metal.scale_[non_metals_cols.index('mg')]
        mg_batch_original = mg_batch * mg_std + mg_mean

    qt_zero = delta_C_zero / mg_batch_original
    boundary_loss_value = torch.mean(qt_zero ** 2)
    return boundary_loss_value

epochs = 10000
train_r2_scores = []
train_mse_scores = []
total_losses = []
data_losses = []
physics_losses = []
boundary_losses = []
valid_losses = []

total_start_time = time.time()
for epoch in range(epochs):
    epoch_start_time = time.time()
    total_loss_epoch = 0.0
    data_loss_epoch = 0.0
    data_loss_qt_epoch = 0.0
    data_loss_Ct_epoch = 0.0
    physics_loss_epoch = 0.0
    boundary_loss_epoch = 0.0

    deeponet_model.train()
    for batch_idx, (branch_batch, trunk_batch, output_qt_batch, output_Ct_batch, C0_batch) in enumerate(train_loader):
        branch_batch = branch_batch.to(device)
        trunk_batch = trunk_batch.to(device)
        output_qt_batch = output_qt_batch.to(device)
        output_Ct_batch = output_Ct_batch.to(device)
        C0_batch = C0_batch.to(device)

        mg_batch = branch_batch[:, mg_index:mg_index + 1]

        trunk_batch.requires_grad_(True)
        delta_C_predictions, qe_predictions, k2_predictions, Ct_predictions = deeponet_model(branch_batch, trunk_batch, C0_batch)

        # 反标准化 mg
        if branch_columns[mg_index] in group_a:
            mg_batch_original = mg_batch * factor_a
        elif branch_columns[mg_index] in group_b:
            mg_batch_original = mg_batch * factor_b
        else:
            mg_mean = branch_scaler_non_metal.mean_[non_metals_cols.index('mg')]
            mg_std = branch_scaler_non_metal.scale_[non_metals_cols.index('mg')]
            mg_batch_original = mg_batch * mg_std + mg_mean

        qt_predictions = delta_C_predictions / mg_batch_original

        data_loss_qt = criterion(qt_predictions, output_qt_batch)
        data_loss_Ct = criterion(Ct_predictions, output_Ct_batch)
        total_data_loss = data_loss_qt + data_loss_Ct

        physics_informed_loss = physics_loss_pseudo_second_order(
            trunk_batch, delta_C_predictions, qe_predictions, k2_predictions,
            trunk_scaler, mg_batch, C0_batch, Ct_predictions, mg_index, branch_scaler_non_metal
        )

        boundary_loss_value = boundary_loss(deeponet_model, branch_batch, C0_batch, mg_index, branch_scaler_non_metal)
        monotonicity_loss_value = monotonicity_loss(
            delta_C_predictions, trunk_batch, trunk_scaler, mg_batch, mg_index, branch_scaler_non_metal
        )

        loss = (deeponet_model.lambda_data * total_data_loss +
                deeponet_model.lambda_physics * physics_informed_loss +
                deeponet_model.lambda_boundary * boundary_loss_value +
                deeponet_model.lambda_monotonicity * monotonicity_loss_value)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(deeponet_model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss_epoch += loss.item()
        data_loss_epoch += total_data_loss.item()
        data_loss_qt_epoch += data_loss_qt.item()
        data_loss_Ct_epoch += data_loss_Ct.item()
        physics_loss_epoch += physics_informed_loss.item()
        boundary_loss_epoch += boundary_loss_value.item()

    total_losses.append(total_loss_epoch / len(train_loader))
    data_losses.append(data_loss_epoch / len(train_loader))
    physics_losses.append(physics_loss_epoch / len(train_loader))
    boundary_losses.append(boundary_loss_epoch / len(train_loader))

    if (epoch + 1) % 10000 == 0:
        checkpoint_path = f'Data7/deeponet_pinn_model_epoch_{epoch + 1}.pth'
        torch.save(deeponet_model.state_dict(), checkpoint_path)
        print(f'已在第 {epoch + 1} 轮保存模型检查点。')

    with torch.no_grad():
        all_train_predictions = []
        all_train_outputs = []
        # 将 lambda 值限制在一定范围内
        deeponet_model.lambda_data.clamp_(min=1)
        deeponet_model.lambda_physics.clamp_(min=0.1)
        deeponet_model.lambda_boundary.clamp_(min=1)
        deeponet_model.lambda_monotonicity.clamp_(min=1)

        for branch_batch, trunk_batch, output_qt_batch, output_Ct_batch, C0_batch in train_loader:
            branch_batch = branch_batch.to(device)
            trunk_batch = trunk_batch.to(device)
            output_qt_batch = output_qt_batch.to(device)
            output_Ct_batch = output_Ct_batch.to(device)
            C0_batch = C0_batch.to(device)

            delta_C_predictions, _, _, Ct_predictions = deeponet_model(branch_batch, trunk_batch, C0_batch)
            mg_batch = branch_batch[:, mg_index:mg_index + 1]

            if branch_columns[mg_index] in group_a:
                mg_batch_original = mg_batch * factor_a
            elif branch_columns[mg_index] in group_b:
                mg_batch_original = mg_batch * factor_b
            else:
                mg_mean = branch_scaler_non_metal.mean_[non_metals_cols.index('mg')]
                mg_std = branch_scaler_non_metal.scale_[non_metals_cols.index('mg')]
                mg_batch_original = mg_batch * mg_std + mg_mean

            qt_predictions = delta_C_predictions / mg_batch_original
            all_train_predictions.append(qt_predictions.cpu().numpy())
            all_train_outputs.append(output_qt_batch.cpu().numpy())

        all_train_predictions = np.concatenate(all_train_predictions)
        all_train_outputs = np.concatenate(all_train_outputs)

        train_mse = mean_squared_error(all_train_outputs, all_train_predictions)
        train_r2 = r2_score(all_train_outputs, all_train_predictions)

        train_mse_scores.append(train_mse)
        train_r2_scores.append(train_r2)

    with torch.no_grad():
        all_val_predictions = []
        all_val_outputs = []
        total_valid_loss = 0.0

        for branch_batch, trunk_batch, output_qt_batch, output_Ct_batch, C0_batch in valid_loader:
            branch_batch = branch_batch.to(device)
            trunk_batch = trunk_batch.to(device)
            output_qt_batch = output_qt_batch.to(device)
            output_Ct_batch = output_Ct_batch.to(device)
            C0_batch = C0_batch.to(device)

            delta_C_predictions, _, _, Ct_predictions = deeponet_model(branch_batch, trunk_batch, C0_batch)
            mg_batch = branch_batch[:, mg_index:mg_index + 1]

            if branch_columns[mg_index] in group_a:
                mg_batch_original = mg_batch * factor_a
            elif branch_columns[mg_index] in group_b:
                mg_batch_original = mg_batch * factor_b
            else:
                mg_mean = branch_scaler_non_metal.mean_[non_metals_cols.index('mg')]
                mg_std = branch_scaler_non_metal.scale_[non_metals_cols.index('mg')]
                mg_batch_original = mg_batch * mg_std + mg_mean

            qt_predictions = delta_C_predictions / mg_batch_original
            all_val_predictions.append(qt_predictions.cpu().numpy())
            all_val_outputs.append(output_qt_batch.cpu().numpy())
            data_loss = criterion(qt_predictions, output_qt_batch)
            total_valid_loss += data_loss.item()

        all_val_predictions = np.concatenate(all_val_predictions)
        all_val_outputs = np.concatenate(all_val_outputs)

        val_mse = mean_squared_error(all_val_outputs, all_val_predictions)
        val_r2 = r2_score(all_val_outputs, all_val_predictions)
        valid_losses.append(total_valid_loss / len(valid_loader))

        if (epoch + 1) % 10 == 0:
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            print(
                f'第 [{epoch + 1}/{epochs}] 轮，损失: {total_loss_epoch / len(train_loader):.4f}, '
                f'数据损失: {data_loss_epoch / len(train_loader):.4f}, '
                f'qt 数据损失: {data_loss_qt_epoch / len(train_loader):.4f}, '
                f'Ct 数据损失: {data_loss_Ct_epoch / len(train_loader):.4f}, '
                f'物理损失: {physics_loss_epoch / len(train_loader):.4f}, '
                f'边界损失: {boundary_loss_epoch / len(train_loader):.4f}, '
                f'训练 MSE: {train_mse:.4f}, 训练 R²: {train_r2:.4f}, '
                f'验证 MSE: {val_mse:.4f}, 验证 R²: {val_r2:.4f}, '
                f'Lambda 数据: {deeponet_model.lambda_data.item():.4f}, '
                f'Lambda 物理: {deeponet_model.lambda_physics.item():.4f}, '
                f'Lambda 边界: {deeponet_model.lambda_boundary.item():.4f}, '
                f'耗时: {epoch_time:.2f} 秒'
            )

torch.save(deeponet_model.state_dict(), model_save_path)

save_dir = './data8'
os.makedirs(save_dir, exist_ok=True)

def evaluate_model(model, dataset, batch_size, branch_scaler, dataset_name, mg_index):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_qt_predictions = []
    all_Ct_predictions = []
    all_qt_true = []
    all_Ct_true = []

    with torch.no_grad():
        for branch_inputs, trunk_inputs, true_qt_outputs, true_Ct_outputs, C0_values in data_loader:
            branch_inputs = branch_inputs.to(device)
            trunk_inputs = trunk_inputs.to(device)
            true_qt_outputs = true_qt_outputs.to(device)
            true_Ct_outputs = true_Ct_outputs.to(device)
            C0_values = C0_values.to(device)

            delta_C_predictions, _, _, Ct_predictions = model(branch_inputs, trunk_inputs, C0_values)

            mg_batch = branch_inputs[:, mg_index:mg_index + 1]
            if branch_columns[mg_index] in group_a:
                mg_batch_original = mg_batch * factor_a
            elif branch_columns[mg_index] in group_b:
                mg_batch_original = mg_batch * factor_b
            else:
                mg_mean = branch_scaler_non_metal.mean_[non_metals_cols.index('mg')]
                mg_std = branch_scaler_non_metal.scale_[non_metals_cols.index('mg')]
                mg_batch_original = mg_batch * mg_std + mg_mean

            qt_predictions = delta_C_predictions / mg_batch_original

            all_qt_predictions.append(qt_predictions.cpu().numpy())
            all_Ct_predictions.append(Ct_predictions.cpu().numpy())
            all_qt_true.append(true_qt_outputs.cpu().numpy())
            all_Ct_true.append(true_Ct_outputs.cpu().numpy())

    mse_qt = mean_squared_error(np.concatenate(all_qt_true), np.concatenate(all_qt_predictions))
    mae_qt = mean_absolute_error(np.concatenate(all_qt_true), np.concatenate(all_qt_predictions))
    r2_qt = r2_score(np.concatenate(all_qt_true), np.concatenate(all_qt_predictions))

    mse_Ct = mean_squared_error(np.concatenate(all_Ct_true), np.concatenate(all_Ct_predictions))
    mae_Ct = mean_absolute_error(np.concatenate(all_Ct_true), np.concatenate(all_Ct_predictions))
    r2_Ct = r2_score(np.concatenate(all_Ct_true), np.concatenate(all_Ct_predictions))

    print(f"{dataset_name} qt - MSE: {mse_qt:.4f}, MAE: {mae_qt:.4f}, R²: {r2_qt:.4f}")
    print(f"{dataset_name} Ct - MSE: {mse_Ct:.4f}, MAE: {mae_Ct:.4f}, R²: {r2_Ct:.4f}")

    return mse_qt, mae_qt, r2_qt, mse_Ct, mae_Ct, r2_Ct, all_qt_predictions, all_qt_true, all_Ct_predictions, all_Ct_true

def save_evaluation_results(model, dataset, batch_size, output_path_qt, output_path_Ct, csv_path, dataset_name,
                            branch_scaler, mg_index):
    print("Using these branch features for prediction:", branch_columns)
    print("Using these trunk features for prediction:", trunk_columns)

    mse_qt, mae_qt, r2_qt, mse_Ct, mae_Ct, r2_Ct, qt_predictions, qt_true, Ct_predictions, Ct_true = evaluate_model(
        model, dataset, batch_size, branch_scaler, dataset_name, mg_index
    )

    qt_true_values = np.concatenate(qt_true, axis=0).reshape(-1)
    qt_predictions = np.concatenate(qt_predictions, axis=0).reshape(-1)
    Ct_true_values = np.concatenate(Ct_true, axis=0).reshape(-1)
    Ct_predictions = np.concatenate(Ct_predictions, axis=0).reshape(-1)

    torch.save(qt_predictions, output_path_qt)
    torch.save(Ct_predictions, output_path_Ct)

    results_df = pd.DataFrame({
        'True qt': qt_true_values,
        'Predicted qt': qt_predictions,
        'True Ct': Ct_true_values,
        'Predicted Ct': Ct_predictions,
    })
    results_df.to_csv(csv_path, index=False)

    print(f"{dataset_name} 结果已保存到 {csv_path}")

save_evaluation_results(
    deeponet_model, train_dataset, batch_size,
    train_pred_save_path_qt, train_pred_save_path_Ct,
    'train_results.csv', '训练集', None, mg_index
)
save_evaluation_results(
    deeponet_model, valid_dataset, batch_size,
    val_pred_save_path_qt, val_pred_save_path_Ct,
    'valid_results.csv', '验证集', None, mg_index
)
save_evaluation_results(
    deeponet_model, test_dataset, batch_size,
    test_pred_save_path_qt, test_pred_save_path_Ct,
    'test_results.csv', '测试集', None, mg_index
)

def plot_losses(total_losses, valid_losses):
    epochs_range = range(1, len(total_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, total_losses, label='Training Loss', color='blue')
    plt.plot(epochs_range, valid_losses, label='Validation Loss', color='red')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 10000)
    plt.savefig('data8/loss_plot.png')
    plt.show()

plot_losses(total_losses, valid_losses)

def plot_metrics(epochs, mse_scores, r2_scores, filename):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1, 1), mse_scores, label='Training MSE', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Training MSE Over Epochs')
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1, 1), r2_scores, label='Training R²', color='orange')
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
    true_values = np.array(true_values).reshape(-1)
    predicted_values = np.array(predicted_values).reshape(-1)

    if true_values.shape[0] != predicted_values.shape[0]:
        raise ValueError("True values and predicted values must have the same length.")

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
    if os.path.exists(train_pred_save_path_qt):
        train_true_qt = outputs_qt_tensor.cpu().numpy().reshape(-1)
        train_pred_qt = torch.load(train_pred_save_path_qt).reshape(-1)
        plot_predictions(train_true_qt, train_pred_qt, "Training Set qt: Predicted vs True", "train_predictions_qt")

    if os.path.exists(train_pred_save_path_Ct):
        train_true_Ct = outputs_Ct_tensor.cpu().numpy().reshape(-1)
        train_pred_Ct = torch.load(train_pred_save_path_Ct).reshape(-1)
        plot_predictions(train_true_Ct, train_pred_Ct, "Training Set Ct: Predicted vs True", "train_predictions_Ct")

    if os.path.exists(val_pred_save_path_qt):
        val_true_qt = val_outputs_qt_tensor.cpu().numpy().reshape(-1)
        val_pred_qt = torch.load(val_pred_save_path_qt).reshape(-1)
        plot_predictions(val_true_qt, val_pred_qt, "Validation Set qt: Predicted vs True", "val_predictions_qt")

    if os.path.exists(val_pred_save_path_Ct):
        val_true_Ct = val_outputs_Ct_tensor.cpu().numpy().reshape(-1)
        val_pred_Ct = torch.load(val_pred_save_path_Ct).reshape(-1)
        plot_predictions(val_true_Ct, val_pred_Ct, "Validation Set Ct: Predicted vs True", "val_predictions_Ct")

    if os.path.exists(test_pred_save_path_qt):
        test_true_qt = test_outputs_qt_tensor.cpu().numpy().reshape(-1)
        test_pred_qt = torch.load(test_pred_save_path_qt).reshape(-1)
        plot_predictions(test_true_qt, test_pred_qt, "Test Set qt: Predicted vs True", "test_predictions_qt")

    if os.path.exists(test_pred_save_path_Ct):
        test_true_Ct = test_outputs_Ct_tensor.cpu().numpy().reshape(-1)
        test_pred_Ct = torch.load(test_pred_save_path_Ct).reshape(-1)
        plot_predictions(test_true_Ct, test_pred_Ct, "Test Set Ct: Predicted vs True", "test_predictions_Ct")

try:
    load_and_save_plots()
    print("success")
except Exception as e:
    print(f"Error during plotting: {e}")
