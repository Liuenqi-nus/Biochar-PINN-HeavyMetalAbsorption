import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pickle
from scipy.optimize import differential_evolution, NonlinearConstraint
import os
import json
from scipy.stats import linregress

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 全局列表用于存储评估过的解
evaluated_solutions = []
feasible_solutions = []

# 为多目标评分设定最大值参考（根据实际经验或预设，可调整）
max_qt = 100.0
max_k2 = 1.0

# 时间点范围（多时间点预测）
time_points = np.linspace(10, 200, num=20)

# 路径配置（请根据实际情况修改）
model_save_path = 'Data7/deeponet_pinn_model_epoch_10000.pth'
branch_scaler_path = 'data8/branch_scaler.pkl'
trunk_scaler_path = 'data8/trunk_scaler.pkl'
label_encoders_path = 'label_encoders.pkl'
metal_ion_features_path = 'metal_ion_features.json'
data_file_path = 'data8/数据集0428.csv'  # 用于预测的数据文件

# 检查文件是否存在
required_files = [
    model_save_path,
    branch_scaler_path,
    trunk_scaler_path,
    label_encoders_path,
    metal_ion_features_path,
    data_file_path
]

for file_path in required_files:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到：{file_path}")

##############################
#   1. 模型结构定义
##############################
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

        # 这些 lambda_xxx 如果仅用于训练过程，在推理阶段可不再需要
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

# 载入缩放器、编码器等
with open(branch_scaler_path, 'rb') as f:
    branch_scaler = pickle.load(f)

with open(trunk_scaler_path, 'rb') as f:
    trunk_scaler = pickle.load(f)

with open(label_encoders_path, 'rb') as f:
    label_encoders = pickle.load(f)

with open(metal_ion_features_path, 'r') as f:
    metal_ion_features = json.load(f)

##############################
#   2. 数据预处理函数
##############################
def encode_text_features(data, label_encoders, is_train=True, metal_ion_features=None):
    text_features = data[['atmosphere', 'biomass type', 'methods', 'metal ions']]
    encoded_text_features = []
    encoded_text_feature_names = []

    for col in text_features.columns[:-1]:
        le = label_encoders[col]
        encoded_col = le.transform(text_features[col])
        encoded_text_features.append(encoded_col)
        encoded_text_feature_names.append(f"{col}_encoded")

    metal_feature_data = []
    for metal in text_features['metal ions']:
        if metal in metal_ion_features:
            metal_content = 1
            metal_features = metal_ion_features[metal]
            metal_feature_data.append([metal_content] + metal_features)
        else:
            # 若没有对应信息，则全部填0
            metal_feature_data.append([0] * 7)

    metal_feature_names = [f'metal_{feature}' for feature in ['content', 'radius', 'electronegativity',
                                                              'max_oxidation_state', 'hydration_energy', 'group',
                                                              'period']]
    metal_feature_data = np.array(metal_feature_data)
    metal_feature_df = pd.DataFrame(metal_feature_data, columns=metal_feature_names, index=data.index)

    if encoded_text_features:
        encoded_text_features = np.column_stack(encoded_text_features)
        encoded_text_features_df = pd.DataFrame(encoded_text_features,
                                                columns=encoded_text_feature_names,
                                                index=data.index)
    else:
        encoded_text_features_df = pd.DataFrame(index=data.index)

    processed_data = pd.concat([data, encoded_text_features_df, metal_feature_df], axis=1)
    return encoded_text_features, processed_data, encoded_text_feature_names, metal_feature_names

# 这里需要列出 branch、trunk 所需的列
branch_columns = [
    'T1', 'T2', 'T3', 'pH', 'time1', 'time2',
    'Fe3', 'Fe2', 'Ag', 'Al3', 'Ce3', 'Cu2', 'La3', 'Mn2', 'Mn7', 'Mg2', 'Zn2',
    'BC', 'C0', 'mg'
]
metal_feature_names = [
    'metal_content', 'metal_radius', 'metal_electronegativity',
    'metal_max_oxidation_state', 'metal_hydration_energy', 'metal_group',
    'metal_period'
]
encoded_text_feature_names = [
    'atmosphere_encoded', 'biomass type_encoded', 'methods_encoded'
]
# 把 metal_feature_names 和 encoded_text_feature_names 一并加入到 branch_columns
branch_columns.extend(metal_feature_names)
branch_columns.extend(encoded_text_feature_names)

trunk_columns = ['time3']

branch_input_size = len(branch_columns)
trunk_input_size = len(trunk_columns)

hidden_size = 128
output_size = 4

# 加载模型
deeponet_model = DeepONet(branch_input_size, trunk_input_size, hidden_size, output_size).to(device)
deeponet_model.load_state_dict(torch.load(model_save_path, map_location=device))
deeponet_model.eval()
print("模型已成功加载。")

##############################
#   3. 准备推理数据
##############################
def prepare_input_data(input_df, label_encoders, metal_ion_features):
    _, processed_data, _, _ = encode_text_features(
        input_df, label_encoders, is_train=False, metal_ion_features=metal_ion_features
    )

    missing_columns = set(branch_columns + trunk_columns) - set(processed_data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    branch_inputs = processed_data[branch_columns].values
    trunk_inputs = processed_data[trunk_columns].values

    branch_inputs_scaled = branch_scaler.transform(branch_inputs)
    trunk_inputs_scaled = trunk_scaler.transform(trunk_inputs)

    branch_inputs_tensor = torch.tensor(branch_inputs_scaled, dtype=torch.float32).to(device)
    trunk_inputs_tensor = torch.tensor(trunk_inputs_scaled, dtype=torch.float32).to(device)
    C0_values = processed_data[['C0']].values
    C0_tensor = torch.tensor(C0_values, dtype=torch.float32).to(device)

    return branch_inputs_tensor, trunk_inputs_tensor, C0_tensor, processed_data

data_df = pd.read_csv(data_file_path)
branch_inputs_tensor, trunk_inputs_tensor, C0_tensor, processed_data = prepare_input_data(
    data_df, label_encoders, metal_ion_features
)

##############################
#   4. 约束与目标函数
##############################
current_iteration = 0
current_objective_value = None

def callback_func(xk, convergence):
    global current_iteration
    current_iteration += 1
    print(f"differential_evolution step {current_iteration}: f(x)= {current_objective_value}")
    return False

# 要优化的金属
metals = ['Fe3', 'Fe2', 'Ag', 'Al3', 'Ce3', 'Cu2', 'La3', 'Mn2', 'Mn7', 'Mg2', 'Zn2']

def sum_features_constraint(x):
    """ (可选) 金属总含量 ≤ 0.3 """
    features = x[9:19]  # 这 11 个金属
    sum_features = np.sum(features)
    return sum_features - 0.3

def double_metal_constraint(x):
    """
    恰好 2 个金属 > 0.0002
    => non_zero_count - 2 = 0
    """
    features = x[9:19]
    indicators = (features > 0.0002).astype(float)
    non_zero_count = np.sum(indicators)
    return non_zero_count - 2

def is_feasible_solution(x):
    """
    如果需要“金属总量 ≤ 0.3 + 恰好 2 个金属 >0.0002”则在此判断
    """
    # 注意：double_metal_constraint(x)==0 表示恰好2个
    # sum_features_constraint(x)<=0 表示金属总量<=0.3
    return (
        abs(double_metal_constraint(x)) < 1e-9 and
        sum_features_constraint(x) <= 0
    )

def compute_qt_predictions(x, time_points):
    x = np.array(x)
    x = np.maximum(x, 0.0)  # 不允许负值

    # 将部分参数取整
    x[0] = np.round(x[0])  # T1
    x[1] = np.round(x[1])  # T2
    x[4] = np.round(x[4])  # time1
    x[5] = np.round(x[5])  # time2

    # 编码值裁剪到合理范围
    atmosphere_encoded = int(x[-3])
    biomass_type_encoded = int(x[-2])
    methods_encoded = int(x[-1])

    max_atmosphere = len(label_encoders['atmosphere'].classes_) - 1
    max_biomass = len(label_encoders['biomass type'].classes_) - 1
    max_methods = len(label_encoders['methods'].classes_) - 1

    atmosphere_encoded = np.clip(atmosphere_encoded, 0, max_atmosphere)
    atmosphere_decoded = label_encoders['atmosphere'].inverse_transform([atmosphere_encoded])[0]

    biomass_type_encoded = np.clip(biomass_type_encoded, 0, max_biomass)
    biomass_type_decoded = label_encoders['biomass type'].inverse_transform([biomass_type_encoded])[0]
    methods_encoded = np.clip(methods_encoded, 0, max_methods)
    methods_decoded = label_encoders['methods'].inverse_transform([methods_encoded])[0]

    # metal ions 固定设置为 'Sb'，仅做示例（您也可改成其它）
    metal_ions = 'Sb'

    input_dict = {
        'T1': [x[0]] * len(time_points),
        'T2': [x[1]] * len(time_points),
        'T3': [x[2]] * len(time_points),
        'pH': [x[3]] * len(time_points),
        'time1': [x[4]] * len(time_points),
        'time2': [x[5]] * len(time_points),
        'time3': list(time_points),
        'C0': [x[7]] * len(time_points),
        'mg': [x[8]] * len(time_points),
        'Fe3': [x[9]] * len(time_points),
        'Fe2': [x[10]] * len(time_points),
        'Ag': [x[11]] * len(time_points),
        'Al3': [x[12]] * len(time_points),
        'Ce3': [x[13]] * len(time_points),
        'Cu2': [x[14]] * len(time_points),
        'La3': [x[15]] * len(time_points),
        'Mn2': [x[16]] * len(time_points),
        'Mn7': [x[17]] * len(time_points),
        'Mg2': [x[18]] * len(time_points),
        'Zn2': [x[19]] * len(time_points),
        'BC': [x[20]] * len(time_points),
        'atmosphere': [atmosphere_decoded] * len(time_points),
        'biomass type': [biomass_type_decoded] * len(time_points),
        'methods': [methods_decoded] * len(time_points),
        'metal ions': [metal_ions] * len(time_points),
    }

    input_df = pd.DataFrame(input_dict)
    _, processed_input_data, _, _ = encode_text_features(
        input_df, label_encoders, is_train=False, metal_ion_features=metal_ion_features
    )

    branch_input = processed_input_data[branch_columns].values
    trunk_input = processed_input_data[trunk_columns].values

    branch_input_scaled = branch_scaler.transform(branch_input)
    trunk_input_scaled = trunk_scaler.transform(trunk_input)

    branch_input_tensor = torch.tensor(branch_input_scaled, dtype=torch.float32).to(device)
    trunk_input_tensor = torch.tensor(trunk_input_scaled, dtype=torch.float32).to(device)
    C0_tensor = torch.tensor([[x[7]]] * len(time_points), dtype=torch.float32).to(device)

    mg_index = branch_columns.index('mg')
    mg_value = branch_input_tensor[:, mg_index]
    mg_mean = torch.tensor(branch_scaler.mean_[mg_index], device=device)
    mg_std = torch.tensor(branch_scaler.scale_[mg_index], device=device)
    mg_value_original = mg_value * mg_std + mg_mean

    with torch.no_grad():
        delta_C_prediction, _, _, _ = deeponet_model(branch_input_tensor, trunk_input_tensor, C0_tensor)

    # 计算 qt = (delta_C_prediction) / mg
    qt_predictions = (delta_C_prediction.squeeze(1) / mg_value_original).cpu().numpy()
    return qt_predictions.tolist()

def base_objective_function(x):
    qt_predictions = compute_qt_predictions(x, time_points)

    # 检查重复度
    qt_std = np.std(qt_predictions)
    redundancy_threshold = 1e-3
    if qt_std < redundancy_threshold:
        return 1e6  # 罚一个大值

    # 取 time3=200 时的 qt 作为 qe
    qt_final = qt_predictions[-1]
    qe = qt_final

    # 根据伪二级动力学：t/qt vs. t => 斜率 => k2
    t_over_qt = np.array(time_points) / np.array(qt_predictions)
    slope, intercept, r_value, p_value, std_err = linregress(time_points, t_over_qt)
    k2 = 1.0 / (slope * (qe ** 2))

    # 归一化
    qt_normalized = qt_final / max_qt
    k2_normalized = k2 / max_k2
    score = 0.5 * qt_normalized + 0.5 * k2_normalized

    # 记录
    evaluated_solutions.append((qt_final, x.copy(), k2, score))

    # 若满足约束且 qt ∈ [80,100]，则添加到 feasible_solutions
    if is_feasible_solution(x) and 80 <= qt_final <= 100:
        feasible_solutions.append((qt_final, x.copy(), k2, score))

    # 返回负数 => 最大化score
    return -score

def combined_objective_function(x):
    global current_objective_value
    val = base_objective_function(x)
    current_objective_value = val
    return val

##############################
#   5. 设置约束并进行优化
##############################
sum_constraint = NonlinearConstraint(sum_features_constraint, -np.inf, 0)
double_metal_cnt_constraint = NonlinearConstraint(double_metal_constraint, 0, 0)

parameter_names = [
    'T1', 'T2', 'T3', 'pH', 'time1', 'time2', 'time3', 'C0', 'mg',
    'Fe3', 'Fe2', 'Ag', 'Al3', 'Ce3', 'Cu2', 'La3', 'Mn2', 'Mn7', 'Mg2', 'Zn2', 'BC',
    'atmosphere_encoded', 'biomass_type_encoded', 'methods_encoded'
]

result = differential_evolution(
    combined_objective_function,
    bounds=[
        (300, 900),  # T1
        (25, 210),   # T2
        (25, 25),    # T3 (固定)
        (4, 7),      # pH
        (0, 100),    # time1
        (0, 100),    # time2
        (200, 200),  # time3 (固定)
        (100, 100),  # C0 (固定)
        (1, 1),      # mg (固定)
        (0, 0.1),    # Fe3
        (0, 0.1),    # Fe2
        (0, 0.1),    # Ag
        (0, 0.1),    # Al3
        (0, 0.1),    # Ce3
        (0, 0.1),    # Cu2
        (0, 0.1),    # La3
        (0, 0.1),    # Mn2
        (0, 0.1),    # Mn7
        (0, 0.1),    # Mg2
        (0, 0),    # Zn2
        (0.005, 1),  # BC
        (0, len(label_encoders['atmosphere'].classes_) - 1),
        (0, len(label_encoders['biomass type'].classes_) - 1),
        (0, len(label_encoders['methods'].classes_) - 1)
    ],
    strategy='rand1bin',
    maxiter=2000,
    popsize=20,
    tol=0.01,
    mutation=(0.5, 1.0),
    recombination=0.5,
    seed=9,
    disp=True,
    polish=True,
    callback=callback_func,
    # 关键：两个约束
    constraints=(sum_constraint, double_metal_cnt_constraint)
)

optimized_parameters = result.x
# 将部分参数取整
optimized_parameters[0] = np.round(optimized_parameters[0])  # T1
optimized_parameters[1] = np.round(optimized_parameters[1])  # T2
optimized_parameters[4] = np.round(optimized_parameters[4])  # time1
optimized_parameters[5] = np.round(optimized_parameters[5])  # time2
optimized_parameters[-3:] = np.round(optimized_parameters[-3:])  # 三个编码变量

atmosphere_encoded = int(optimized_parameters[-3])
biomass_type_encoded = int(optimized_parameters[-2])
methods_encoded = int(optimized_parameters[-1])

# clip 到合理范围
atmosphere_encoded = np.clip(atmosphere_encoded, 0, len(label_encoders['atmosphere'].classes_) - 1)
biomass_type_encoded = np.clip(biomass_type_encoded, 0, len(label_encoders['biomass type'].classes_) - 1)
methods_encoded = np.clip(methods_encoded, 0, len(label_encoders['methods'].classes_) - 1)

atmosphere_decoded = label_encoders['atmosphere'].inverse_transform([atmosphere_encoded])[0]
biomass_type_decoded = label_encoders['biomass type'].inverse_transform([biomass_type_encoded])[0]
methods_decoded = label_encoders['methods'].inverse_transform([methods_encoded])[0]

final_score = -result.fun

print("\n优化的参数值（最终）：")
for name, value in zip(parameter_names[:-3], optimized_parameters[:-3]):
    print(f"{name}: {value}")
print(f"atmosphere: {atmosphere_decoded}")
print(f"biomass type: {biomass_type_decoded}")
print(f"methods: {methods_decoded}")
print(f"\n最大综合评分值: {final_score}")

# 保存到CSV
output_dict = {name: [value] for name, value in zip(parameter_names[:-3], optimized_parameters[:-3])}
output_dict['atmosphere'] = atmosphere_decoded
output_dict['biomass type'] = biomass_type_decoded
output_dict['methods'] = methods_decoded
output_dict['score'] = final_score

output_df = pd.DataFrame(output_dict)
output_df.to_csv('optimized_result.csv', index=False)
print("\n优化结果已保存到 optimized_result.csv")

##############################
#   6. 导出可行解
##############################
def compute_qt_predictions_for_solution(x):
    return compute_qt_predictions(x, time_points)

if len(feasible_solutions) == 0:
    print("没有找到满足约束条件且 qt ∈ [80, 100] 的可行解。")
else:
    # 按 score 倒序
    feasible_solutions.sort(key=lambda kv: kv[3], reverse=True)
    top_100_solutions = feasible_solutions[:10000]

    rows = []
    for idx, (qt_val, x_val, k2_val, score_val) in enumerate(top_100_solutions, start=1):
        qt_preds = compute_qt_predictions_for_solution(x_val)

        # 检查最后一点
        if not np.isclose(qt_val, qt_preds[-1], atol=1e-6):
            print(f"警告：解 {idx} 的 qt_final ({qt_val}) 与 qt_prediction_200 ({qt_preds[-1]}) 不一致。")

        qt_std = np.std(qt_preds)
        redundancy_threshold = 1e-3
        if qt_std < redundancy_threshold:
            print(f"解 {idx} 被跳过，因为标准差 ({qt_std}) 过低。")
            continue

        # 解码
        atm_enc = int(np.round(x_val[-3]))
        bio_enc = int(np.round(x_val[-2]))
        mth_enc = int(np.round(x_val[-1]))

        atm_dec = label_encoders['atmosphere'].inverse_transform([atm_enc])[0]
        bio_dec = label_encoders['biomass type'].inverse_transform([bio_enc])[0]
        mth_dec = label_encoders['methods'].inverse_transform([mth_enc])[0]

        row = {name: val for name, val in zip(parameter_names, x_val)}
        row['atmosphere'] = atm_dec
        row['biomass type'] = bio_dec
        row['methods'] = mth_dec

        # 一些关键指标
        row['qt_prediction'] = qt_val
        row['k2'] = k2_val
        row['score'] = score_val
        row['metal ions'] = 'Sb'  # 因为我们在 compute_qt_predictions 里写死了

        # 多时间点输出
        for t, qt_p in zip(time_points, qt_preds):
            row[f'qt_prediction_{int(t)}'] = qt_p

        rows.append(row)

    top_100_df = pd.DataFrame(rows)

    desired_columns_order = [
        'biomass type', 'T1', 'time1', 'atmosphere', 'methods',
        'Fe3', 'Fe2', 'Ag', 'Al3', 'Ce3', 'Cu2', 'La3', 'Mn2', 'Mn7', 'Mg2', 'Zn2',
        'BC', 'time2', 'T2', 'metal ions', 'C0', 'mg', 'pH', 'time3', 'T3',
        'qt_prediction', 'k2', 'score'
    ]
    qt_time_columns = [f'qt_prediction_{int(t)}' for t in time_points]
    desired_columns_order.extend(qt_time_columns)

    # 如果缺列，填充 NaN
    missing_columns = set(desired_columns_order) - set(top_100_df.columns)
    for col in missing_columns:
        top_100_df[col] = np.nan

    top_100_df = top_100_df[desired_columns_order]
    top_100_df.to_csv('top_100_solutions_multiobjective.csv', index=False)
    print("前100组可行解已保存到 top_100_solutions_multiobjective.csv，并包含多时间点 qt 预测结果。")
