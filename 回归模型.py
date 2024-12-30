import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
# 加载数据集
df = pd.read_excel('数据集0428.xlsx')
# 对df的所有数值型列进行遍历
for col in df.columns:
    if df[col].dtype == 'float64':  # 检查数据类型
        df[col] = df[col].round(2)  # 将数值型列的精度修改为2位小数

# 定义函数：映射原始特征到独热编码后的特征
def map_original_features_to_encoded(df_encoded, original_features):
    feature_map = {}
    for feature in original_features:
        # 找到所有以原始特征名称开头的列
        encoded_features = [col for col in df_encoded.columns if col.startswith(feature)]
        feature_map[feature] = encoded_features
    return feature_map

# 定义函数：计算每个原始特征组的整体特征重要性
def calculate_feature_group_importance(feature_importances, encoded_feature_map, df_encoded, numeric_features):
    feature_group_importance = {}
    for original_feature, encoded_features in encoded_feature_map.items():
        # 确认索引有效，避免索引越界
        indices = [index for index in df_encoded.columns.get_indexer(encoded_features) if index >= 0]
        total_importance = sum(feature_importances[indices])
        feature_group_importance[original_feature] = total_importance

    # 处理数值型特征
    for numeric_feature in numeric_features:
        if numeric_feature in df_encoded.columns:
            index = df_encoded.columns.get_loc(numeric_feature)
            feature_group_importance[numeric_feature] = feature_importances[index]

    return feature_group_importance


# 定义数值型特征列表
numeric_features = [col for col in df.columns if col != 'adsorption efficiency (%)'][:24]

# 应用独热编码
columns_to_encode = ['biomass name', 'biomass type', 'atmosphere', 'methods', 'M1', 'M2', 'M3', 'F1', 'F2', 'C1', 'C2', 'C3', 'metal ions']
df_encoded = pd.get_dummies(df, columns=columns_to_encode)



# 分离特征和目标变量
X = df_encoded.drop('adsorption efficiency (%)', axis=1)  # 使用正确的目标变量列名
y = df_encoded['adsorption efficiency (%)']

# 创建映射
encoded_feature_map = map_original_features_to_encoded(X, columns_to_encode)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建随机森林模型
param_grid = {
    'n_estimators': [100, 500, 1000],  # 树的数量
    'max_depth': [None, 10, 20],  # 树的最大深度
    'min_samples_split': [2, 5, 10],  # 分裂内部节点所需的最小样本数
    'min_samples_leaf': [1, 2, 4],  # 叶节点所需的最小样本数
    'max_features': ['auto', 'sqrt']  # 寻找最佳分裂时要考虑的特征数量
}

# 创建决策树回归模型的网格搜索
# param_grid = {
#     'max_depth': [None, 10, 20, 30],  # 树的最大深度
#     'min_samples_split': [2, 5, 10],  # 分裂内部节点所需的最小样本数
#     'min_samples_leaf': [1, 2, 4],  # 叶节点所需的最小样本数
#     'max_features': ['auto', 'sqrt', 'log2']  # 寻找最佳分裂时要考虑的特征数量
# }

# param_grid = {
#     'C': [0.1, 1, 10],  # 正则化参数
#     'gamma': ['scale', 'auto'],  # 核函数的系数
#     'kernel': ['linear', 'rbf', 'poly']  # 使用的核函数
# }

# 创建支持向量机回归模型的网格搜索
grid_search = GridSearchCV(estimator=DecisionTreeRegressor(),
                           param_grid=param_grid,
                           cv=3,
                           n_jobs=1,
                           verbose=2,
                           scoring='neg_mean_squared_error')
# DecisionTreeRegressor
# RandomForestRegressor
# 创建GridSearchCV实例


# 对缩放后的训练数据进行网格搜索
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print("Best parameters:", grid_search.best_params_)

# # 计算每个原始特征组的整体特征重要性
# if grid_search.best_params_['kernel'] == 'linear':
#     best_model = grid_search.best_estimator_
#     feature_weights = best_model.coef_
#     # 输出特征权重
#     for i, weight in enumerate(feature_weights[0]):
#         print(f"Feature: {X.columns[i]}, Weight: {weight}")
# else:
#     perm_importance = permutation_importance(best_model, X_test_scaled, y_test, n_repeats=30, random_state=42)
#
#     feature_names = X.columns.tolist()
#     importance_sorted_idx = np.argsort(perm_importance.importances_mean)[::-1]
#     for idx in importance_sorted_idx:
#         print(f"Feature: {feature_names[idx]}, Importance: {perm_importance.importances_mean[idx]:.3f}")

feature_importances = best_model.feature_importances_
group_importances = calculate_feature_group_importance(feature_importances, encoded_feature_map, X, numeric_features)

# 打印每个原始特征的总重要性
for feature, importance in group_importances.items():
    print(f"The overall feature importance for '{feature}' is: {importance}")
print("Best parameters:", grid_search.best_params_)

# 获取特征名称和它们的重要性
features = X.columns
importances = best_model.feature_importances_

# 创建特征重要性的DataFrame
feature_importances_df = pd.DataFrame({'Feature': features, 'Importance': importances})

# 对特征重要性进行降序排序
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

# 选择排名前20的最重要特征进行可视化
top_n = 20
top_features = feature_importances_df[:top_n]

# 创建条形图
plt.figure(figsize=(10, 8))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 20 Feature Importance')
plt.gca().invert_yaxis()  # 使得最重要的特征在图的顶部
plt.show()