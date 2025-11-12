# 3_2.py 超参数与流程设置说明

本文件汇总 `code/3_2.py`（以及其直接影响复现的上游预处理）中用于生成“单方法/单模态”谱图与融合结果的关键超参数与流程。旨在保证第三方能够复现实验步骤与结果。

## 数据预处理（上游，供参考）
- Savitzky–Golay 滤波器（来源：`code/data_preprocess.ipynb`）
  - `window_length`: 11
  - `polyorder`: 3
  - 用途：对光谱数据进行平滑与（可选）一阶导数计算
  - 说明：虽然 `3_2.py` 本身未调用 SG 滤波，但若单方法谱图基于预处理数据，则以上参数直接影响复现。

## 特征选择与重要性计算相关
- PLS 回归（VIP）
  - `n_components`: `min(10, 特征数, 样本数-1)`（动态设定）
  - 输入缩放：`StandardScaler()` 对 `X` 与 `y` 分别标准化
- SHAP 解释器选择与采样
  - 优先使用 `shap.TreeExplainer`（树模型）；否则回退为 `shap.KernelExplainer`
  - KernelExplainer 采样：当 `X_used.shape[0] > 100` 时，使用 `shap.sample(X_used, 100)` 作为背景样本
- 统计方法
  - 相关系数：Spearman、Pearson（无额外超参数）
  - dCor：`dcor.distance_correlation`（无额外超参数）
  - HSIC：脚本内自定义实现（标准化 + 核矩阵 + 居中矩阵），无外部库超参数
- 排名计算
  - `pandas.Series.rank(ascending=False, method="min")`（用于各方法的重要性排名）

## 机器学习模型（用于特征重要性/评分）
- 数据划分
  - `train_test_split(test_size=0.2, random_state=42)`（两处使用）
- 随机森林（RandomForestRegressor）
  - `n_estimators=200`, `random_state=42`
- XGBoost（XGBRegressor）
  - `n_estimators=200`, `random_state=42`, `verbosity=0`
- LightGBM（LGBMRegressor）
  - `n_estimators=200`, `random_state=42`, `verbose=-1`
- CatBoost（CatBoostRegressor）
  - 变体一：`iterations=200, depth=6, learning_rate=0.1, random_state=42, verbose=0`
  - 变体二：`n_estimators=200, random_state=42, verbose=0`
  - 说明：两处出现不同构造方式，均对应约 200 次提升迭代；一处显式设定 `depth` 与 `learning_rate`
- KNN（KNeighborsRegressor）
  - `n_neighbors=5`
- SVR（支持向量回归）
  - `kernel="rbf"`, `C=1.0`, `epsilon=0.1`
- MLP（MLPRegressor）
  - `hidden_layer_sizes=(100, 50)`, `max_iter=500`, `random_state=42`

## 融合评价与权重设置
- NMF 共识曲线
  - `n_components=1`, `init="nndsvd"`, `random_state=0`, `max_iter=1000`
  - 非负化：对评分矩阵 `X` 做移位确保非负（`X_nonneg = X - X.min() + 1e-6`）
- 三指标（方法一致性、平滑度、成段性）
  - 归一化：若列和不为 0，则按列和归一化；否则均分
  - 平滑度：`smoothness_score` 使用总变差的反函数（无外部超参数）
  - 成段性（局部能量比）：`window_size=10`
- 动态融合权重（当未显式传入 `weights` 时）
  - 可分性度量：三指标的列标准差
  - Softmax 温度：`softmax_tau=5.0`
  - 与均匀分布的平滑系数：`dynamic_alpha=0.3`
  - 说明：最终权重 `w_final = (1.0 - dynamic_alpha) * uniform + dynamic_alpha * softmax(stds/tau)`
- 最终融合分数归一化：对 `Final_Weight` 求和后再归一化，避免权重尺度差异

## 交叉验证与随机性控制
- KFold（对原始数据进行折分）
  - `n_splits=5`, `shuffle=True`, `random_state=1`
- 随机数种子
  - 主要设置点：`train_test_split`（42）、各模型（42）、`NMF`（0）、`KFold`（1）
  - 说明：不同组件的随机种子不完全一致，复现时请保持一致设置或根据脚本原值执行

## 绘图与输出
- 输出目录：融合结果默认保存在 `融合结果/`（如热力图、折线图、加权指标等）
- 指标权重导出：`metric_weights.csv`（包含 Consistency/Smoothness/Segmentation 三者的最终权重）

## 版本依赖说明
- 依赖参考：`code/requirements.txt`（含 `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `xgboost`, `lightgbm`, `catboost`, `shap`, `dcor`, `scipy`, `openpyxl` 等）
- 若需要精确复现，请固定依赖的版本号（可在 `requirements.txt` 中补充具体版本）。
