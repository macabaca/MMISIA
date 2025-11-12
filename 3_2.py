import os
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split, KFold

warnings.filterwarnings("ignore")

# ========== 配置 ==========
data_path = "整合数据/final_data_cleaned.xlsx"
target_cols = ["Mad", "Vad"] # 修改为两个目标
sample_col = "Sample"
output_base_dir = "integrated_outputs" # 统一的输出目录
os.makedirs(output_base_dir, exist_ok=True)


# ========== 读取数据 ==========
df_original = pd.read_excel(data_path)
exclude_cols = [sample_col, "Mad", "Vad"]
feature_cols = [c for c in df_original.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_original[c])]

# ========== 数据规模信息 ==========
print(f"原始数据维度: {df_original.shape}")

import os
import pandas as pd
import numpy as np
import shap
import dcor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm.auto import tqdm

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import NMF
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")
# ========== 定义HSIC函数和Partial Corr函数 (保持不变) ==========
def hsic(x, y):
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    K = np.outer(x, x)
    L = np.outer(y, y)
    H = np.eye(len(x)) - 1.0 / len(x)
    return np.trace(K @ H @ L @ H) / ((len(x) - 1) ** 2)

def partial_corr(X, y, idx):
    xi = X[:, idx]
    X_rest = np.delete(X, idx, axis=1)
    # 确保 X_rest 不为空
    if X_rest.shape[1] == 0:
        return spearmanr(xi, y)[0] # 如果没有其他特征，直接计算xi和y的相关性
    model_x = LinearRegression().fit(X_rest, xi)
    res_x = xi - model_x.predict(X_rest)
    model_y = LinearRegression().fit(X_rest, y)
    res_y = y - model_y.predict(X_rest)
    rho, _ = spearmanr(res_x, res_y)
    return rho

# ========== 融合评价函数 (原第二段代码) ==========
# 1. 平滑度 & 成段性指标
def smoothness_score(curve):
    tv = np.sum(np.abs(np.diff(curve)))
    return 1 / (1 + tv)

def segmentation_score(curve, window_size=10):
    energy_total = np.sum(curve**2)
    if energy_total == 0:
        return 0
    max_ratio = 0
    for i in range(len(curve) - window_size + 1):
        window_energy = np.sum(curve[i:i+window_size]**2)
        ratio = window_energy / energy_total
        if ratio > max_ratio:
            max_ratio = ratio
    return max_ratio

# --- 三个指标加权融合
def compute_final_weights(results_df, weights=None, dynamic_alpha=0.3, softmax_tau=5.0):
    """
    根据三个指标结果动态分配融合权重：
    - 如果传入 weights(dict)，则使用显式权重（与原逻辑一致）。
    - 否则采用“标准差 → Softmax → 与均匀分布平滑”方案，避免权重差距过大。

    参数：
    - dynamic_alpha: 与均匀分布的平滑系数，越小越接近均匀，默认 0.3。
    - softmax_tau: Softmax 温度（越大越平滑、差距越小），默认 5.0。
    """
    cols = ["NMF_Weight", "Smoothness", "Segmentation"]

    if weights is not None:
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        # 保存归一化后的显式权重
        w_consistency = weights.get("NMF_Weight", 0.0)
        w_smoothness = weights.get("Smoothness", 0.0)
        w_segmentation = weights.get("Segmentation", 0.0)

        results_df["Final_Weight"] = (
            w_consistency * results_df["NMF_Weight"] +
            w_smoothness * results_df["Smoothness"] +
            w_segmentation * results_df["Segmentation"]
        )
        # 将权重与加权后的分量保存到结果中（每行相同，便于后续绘图）
        results_df["Weight_Consistency"] = w_consistency
        results_df["Weight_Smoothness"] = w_smoothness
        results_df["Weight_Segmentation"] = w_segmentation
        results_df["NMF_Weight_Weighted"] = results_df["NMF_Weight"] * w_consistency
        results_df["Smoothness_Weighted"] = results_df["Smoothness"] * w_smoothness
        results_df["Segmentation_Weighted"] = results_df["Segmentation"] * w_segmentation
    else:
        # 1) 指标的“可分性”：使用列标准差衡量（越大说明区分能力越强）
        stds = results_df[cols].std().values.astype(float)
        eps = 1e-12
        stds = stds + eps

        # 2) Softmax（带温度）得到动态权重，避免极端值
        w_dyn = np.exp(stds / max(softmax_tau, eps))
        w_dyn = w_dyn / w_dyn.sum()

        # 3) 与均匀分布做凸组合，限制差距不要太大
        uniform = np.ones_like(w_dyn) / len(w_dyn)
        w_final = (1.0 - dynamic_alpha) * uniform + dynamic_alpha * w_dyn
        # 拆解三个指标的最终权重
        w_consistency, w_smoothness, w_segmentation = float(w_final[0]), float(w_final[1]), float(w_final[2])

        # 4) 应用到每个方法上：加权求和得到最终融合分数
        results_df["Final_Weight"] = results_df[cols].values @ w_final
        # 将权重与加权后的分量保存到结果中（每行相同，便于后续绘图）
        results_df["Weight_Consistency"] = w_consistency
        results_df["Weight_Smoothness"] = w_smoothness
        results_df["Weight_Segmentation"] = w_segmentation
        results_df["NMF_Weight_Weighted"] = results_df["NMF_Weight"] * w_consistency
        results_df["Smoothness_Weighted"] = results_df["Smoothness"] * w_smoothness
        results_df["Segmentation_Weighted"] = results_df["Segmentation"] * w_segmentation

    # 归一化，避免除以零
    total_fw = results_df["Final_Weight"].sum()
    if total_fw != 0:
        results_df["Final_Weight"] /= total_fw
    return results_df

# 2. 综合评价函数
def evaluate_methods(df_scores, df_ranks, window_size=10, output_dir="融合结果"):
    wavelengths = df_scores["Wavelength"].values
    methods = df_scores.columns[1:]

    # --- 数据非负化（用于 NMF）
    X = df_scores[methods].values
    X_nonneg = X - X.min() + 1e-6

    # --- NMF 分解
    # 检查 X_nonneg 是否有足够的样本和特征
    if X_nonneg.shape[0] < 2 or X_nonneg.shape[1] < 2:
        print(f"警告: 数据维度不足 ({X_nonneg.shape})，无法进行 NMF 分解。跳过融合。")
        # 返回原始的第一个方法作为融合结果的替代，或抛出错误
        fused_curve = df_scores[methods[0]].values if len(methods) > 0 else np.zeros(len(wavelengths))
        fused_rank = pd.Series(fused_curve).rank(ascending=False, method="min").astype(int)
        
        scores_out = df_scores.copy()
        scores_out["Fused_Score"] = fused_curve
        ranks_out = df_ranks.copy()
        ranks_out["Fused_Rank"] = fused_rank
        
        os.makedirs(output_dir, exist_ok=True)
        scores_out.to_csv(os.path.join(output_dir, "scores_with_fused.csv"), index=False)
        ranks_out.to_csv(os.path.join(output_dir, "ranks_with_fused.csv"), index=False)
        
        plt.figure(figsize=(10, 4))
        plt.plot(wavelengths, fused_curve, color="red", label="Fused Curve (NMF Skipped)")
        plt.title("Final Integrated Importance Curve (NMF Skipped)")
        plt.savefig(os.path.join(output_dir, "fused_curve.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(12, 2))
        heatmap_data = fused_curve.reshape(1, -1)
        plt.imshow(heatmap_data, aspect="auto", cmap="viridis",
                   extent=[wavelengths.min(), wavelengths.max(), 0, 1])
        plt.title("Fused Importance Heatmap (NMF Skipped)")
        plt.savefig(os.path.join(output_dir, "fused_heatmap.png"), dpi=300)
        plt.close()
        
        return scores_out, ranks_out, pd.DataFrame(), fused_curve, wavelengths

    nmf = NMF(n_components=1, init="nndsvd", random_state=0, max_iter=1000)
    W = nmf.fit_transform(X_nonneg)[:, 0]
    H = nmf.components_[0]
    # 方法贡献度（归一化）
    if H.sum() != 0:
        nmf_weights = H / H.sum()
    else:
        nmf_weights = np.ones_like(H) / len(H) # 避免除以零

    # --- 共识曲线归一化（仅用于相关性计算更稳定）
    consensus_curve = (W - W.min()) / (W.max() - W.min() + 1e-9)

    # --- 方法与共识曲线的相关性作为“权重”
    method_corr = {}
    for i, m in enumerate(methods):
        # 确保用于相关性计算的数组至少有两个元素
        if len(X_nonneg[:, i]) > 1 and len(consensus_curve) > 1:
            corr = np.corrcoef(X_nonneg[:, i], consensus_curve)[0, 1]
        else:
            corr = 0 # 如果数据不足，相关性设为0
        method_corr[m] = corr

    # 转成 Series
    method_corr = pd.Series(method_corr)

    # --- 平滑度和局部成段性
    scores = {}
    for i, m in enumerate(methods):
        curve = df_scores[m].values
        # 确保曲线可以归一化，避免除以零
        if curve.max() - curve.min() != 0:
            curve_norm = (curve - curve.min()) / (curve.max() - curve.min() + 1e-9)
        else:
            curve_norm = np.zeros_like(curve) # 如果曲线是常数，归一化为0
        scores[m] = {
            "NMF_Weight": method_corr[i],
            "Smoothness": smoothness_score(curve_norm),
            "Segmentation": segmentation_score(curve_norm, window_size)
        }

    results_df = pd.DataFrame(scores).T

    # --- 三个指标归一化
    for col in ["NMF_Weight", "Smoothness", "Segmentation"]:
        if results_df[col].sum() != 0: # 避免除以零
            results_df[col] = results_df[col] / results_df[col].sum()
        else:
            results_df[col] = 1.0 / len(results_df) # 平均分配

    # --- 最终融合权重
    results_df = compute_final_weights(results_df)

    # --- 得到最终融合曲线
    fused_curve = np.zeros(len(wavelengths))
    for m in methods:
        # 确保权重存在且为有效数字
        weight = results_df.loc[m, "Final_Weight"] if m in results_df.index and pd.notna(results_df.loc[m, "Final_Weight"]) else 0
        fused_curve += df_scores[m].values * weight

    # --- 融合后的排名
    fused_rank = pd.Series(fused_curve).rank(ascending=False, method="min").astype(int)

    # --- 输出两个文件
    os.makedirs(output_dir, exist_ok=True)

    scores_out = df_scores.copy()
    scores_out["Fused_Score"] = fused_curve
    scores_out.to_csv(os.path.join(output_dir, "scores_with_fused.csv"), index=False)

    ranks_out = df_ranks.copy()
    ranks_out["Fused_Rank"] = fused_rank
    ranks_out.to_csv(os.path.join(output_dir, "ranks_with_fused.csv"), index=False)

    # --- 保存三个指标的动态权重（用于后续堆叠图）
    try:
        metric_weights_df = pd.DataFrame({
            "Metric": ["Consistency", "Smoothness", "Segmentation"],
            "Weight": [
                results_df["Weight_Consistency"].iloc[0] if "Weight_Consistency" in results_df.columns else np.nan,
                results_df["Weight_Smoothness"].iloc[0] if "Weight_Smoothness" in results_df.columns else np.nan,
                results_df["Weight_Segmentation"].iloc[0] if "Weight_Segmentation" in results_df.columns else np.nan,
            ]
        })
        metric_weights_df.to_csv(os.path.join(output_dir, "metric_weights.csv"), index=False)
    except Exception as e:
        print(f"警告：保存 metric_weights.csv 失败：{e}")

    # --- 绘制并保存折线图
    plt.figure(figsize=(10, 4))
    plt.plot(wavelengths, fused_curve, color="red", label="Fused Curve")
    plt.xlabel("Wavelength")
    plt.ylabel("Fused Importance")
    plt.title("Final Integrated Importance Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fused_curve.png"), dpi=300)
    plt.close()

    # --- 绘制并保存热力图
    plt.figure(figsize=(12, 2))
    heatmap_data = fused_curve.reshape(1, -1)
    plt.imshow(heatmap_data, aspect="auto", cmap="viridis",
               extent=[wavelengths.min(), wavelengths.max(), 0, 1])
    plt.colorbar(label="Importance")
    plt.xlabel("Wavelength")
    plt.yticks([])
    plt.title("Fused Importance Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fused_heatmap.png"), dpi=300)
    plt.close()

    print(f"✅ 已保存两个文件和两张图片到: {output_dir}")
    return scores_out, ranks_out, results_df, fused_curve, wavelengths


"""
# ===============================================
# 主循环：针对 "Mad" 和 "Vad" 分别进行特征选择和融合（已弃用）
# 说明：该单次划分主循环会使用未定义的 df_train/df_test。
#       现已使用下方的 K 折封装 run_pipeline_for_fold 来执行完整流程。
# ===============================================
all_fused_results = {} # 存储每个目标变量的融合结果

for current_target_col in target_cols:
    print(f"\n======== 正在处理目标变量: {current_target_col} ========\n")

    # 获取训练集数据
    X_train_fs = df_train[feature_cols].values.astype(float)
    y_train_fs = df_train[current_target_col].values.astype(float)

    # ========== 结果存储 (为当前目标变量初始化) ==========
    all_scores = {"Wavelength": feature_cols}
    all_ranks = {"Wavelength": feature_cols}

    # ========== 方法 1: PLS (VIP) ==========
    print("计算 PLS VIP ...")
    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    X_scaled = scaler_x.fit_transform(X_train_fs)
    y_scaled = scaler_y.fit_transform(y_train_fs.reshape(-1, 1)).ravel()

    # 确保 n_components 不超过特征数或样本数-1
    n_components = min(10, X_train_fs.shape[1], X_train_fs.shape[0] - 1)
    if n_components > 0:
        pls = PLSRegression(n_components=n_components)
        pls.fit(X_scaled, y_scaled)

        t = pls.x_scores_
        w = pls.x_weights_
        q = pls.y_loadings_
        if q.shape[0] > 1:
            q = np.sqrt((q ** 2).sum(axis=0))
        else:
            q = q.ravel()
        p, A = X_train_fs.shape[1], t.shape[1]
        s = np.array([(t[:, a] ** 2).sum() * (q[a] ** 2) for a in range(A)])
        # 避免除以零
        if s.sum() != 0:
            vip = np.sqrt(p * ((w ** 2) @ s) / s.sum())
        else:
            vip = np.zeros(p) # 如果s.sum()为0，则vip为0

        all_scores["PLS"] = vip
        all_ranks["PLS_Rank"] = pd.Series(vip).rank(ascending=False, method="min").astype(int)
    else:
        print("警告: 数据不足以进行PLS回归，PLS VIP计算跳过。")
        all_scores["PLS"] = np.zeros(len(feature_cols))
        all_ranks["PLS_Rank"] = pd.Series(np.zeros(len(feature_cols))).rank(ascending=False, method="min").astype(int)


    # ========== 方法 2: 树模型 + SHAP ==========
    print("计算 树模型 + SHAP ...")
    # 这里依然使用df_train的数据进行内部的训练/验证，以计算特征重要性
    X_model_train, X_model_test, y_model_train, y_model_test = train_test_split(
        X_train_fs, y_train_fs, test_size=0.2, random_state=42
    )

    models_fs = {
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=200, random_state=42, verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=200, random_state=42, verbose=-1),
        "CatBoost": CatBoostRegressor(iterations=200, depth=6, learning_rate=0.1,
                                      random_state=42, verbose=0)
    }

    def compute_importance(name, model, X_used, y_train_subset):
        if name in ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]:
            try:
                # 检查模型是否已拟合，避免重复拟合
                if not hasattr(model, "tree_model") and not hasattr(model, "best_iteration_"): # CatBoost特殊处理
                     model.fit(X_used, y_train_subset)
                
                # 如果模型支持 feature_importances_ 属性，优先使用它作为 fallback
                if hasattr(model, 'feature_importances_'):
                    explainer = shap.TreeExplainer(model)
                else: # Fallback to KernelExplainer for models without tree_model or if TreeExplainer fails
                    print(f"警告: 模型 {name} 不支持TreeExplainer或其内部结构不匹配，尝试使用KernelExplainer (较慢)。")
                    # 对于非树模型或无法使用 TreeExplainer 的情况，使用 KernelExplainer
                    # 需要一个小的样本集来提高速度
                    if X_used.shape[0] > 100:
                        X_sampling = shap.sample(X_used, 100) # 从X_used中采样100个样本
                    else:
                        X_sampling = X_used
                    explainer = shap.KernelExplainer(model.predict, X_sampling)
                
                shap_values = explainer.shap_values(X_used)
                
                # 处理多输出模型的情况 (shap_values 可能是列表)
                if isinstance(shap_values, list):
                    feature_importance = np.abs(shap_values[0]).mean(axis=0) # 取第一个输出的SHAP值
                else:
                    feature_importance = np.abs(shap_values).mean(axis=0)
            except Exception as e:
                print(f"错误: 计算 {name} 的SHAP值失败: {e}，尝试使用 model.feature_importances_。")
                if hasattr(model, 'feature_importances_'):
                    feature_importance = model.feature_importances_
                else:
                    print(f"警告: 模型 {name} 也没有 feature_importances_ 属性，将使用 Permutation Importance。")
                    from sklearn.inspection import permutation_importance
                    result = permutation_importance(model, X_used, y_train_subset, n_repeats=10, random_state=42)
                    feature_importance = result.importances_mean
        else:
            from sklearn.inspection import permutation_importance
            result = permutation_importance(model, X_used, y_train_subset, n_repeats=10, random_state=42)
            feature_importance = result.importances_mean
        return feature_importance


    for name, model in models_fs.items():
        # 确保模型在X_model_train上进行训练
        model.fit(X_model_train, y_model_train)
        importance = compute_importance(name, model, X_model_train, y_model_train) # 使用训练集计算SHAP
        all_scores[name] = importance
        all_ranks[name + "_Rank"] = pd.Series(importance).rank(ascending=False, method="min").astype(int)


    # ========== 方法 3: 统计方法 ==========
    print("计算 统计方法 ...")

    # Spearman
    # 确保数据有足够的非NaN值来计算相关性
    spearman_scores = [spearmanr(X_train_fs[:, i], y_train_fs)[0] if len(np.unique(X_train_fs[:, i])) > 1 and len(np.unique(y_train_fs)) > 1 else 0 for i in range(X_train_fs.shape[1])]
    all_scores["Spearman"] = spearman_scores
    all_ranks["Spearman_Rank"] = pd.Series(np.abs(spearman_scores)).rank(ascending=False, method="min").astype(int)

    # Pearson
    print("正在计算 Pearson 相关系数...")
    pearson_scores = [pearsonr(X_train_fs[:, i], y_train_fs)[0] if len(np.unique(X_train_fs[:, i])) > 1 and len(np.unique(y_train_fs)) > 1 else 0 for i in range(X_train_fs.shape[1])]
    all_scores["Pearson"] = pearson_scores
    all_ranks["Pearson_Rank"] = pd.Series(np.abs(pearson_scores)).rank(ascending=False, method="min").astype(int)

    # dCor
    print("正在计算 dCor ...")
    dcor_scores = [dcor.distance_correlation(X_train_fs[:, i], y_train_fs) for i in range(X_train_fs.shape[1])]
    all_scores["dCor"] = dcor_scores
    all_ranks["dCor_Rank"] = pd.Series(dcor_scores).rank(ascending=False, method="min").astype(int)

    # HSIC
    print("正在计算 HSIC ...")
    hsic_scores = [hsic(X_train_fs[:, i], y_train_fs) for i in range(X_train_fs.shape[1])]
    all_scores["HSIC"] = hsic_scores
    all_ranks["HSIC_Rank"] = pd.Series(hsic_scores).rank(ascending=False, method="min").astype(int)

    # MI
    print("正在计算 MI ...")
    mi_scores = mutual_info_regression(X_train_fs, y_train_fs, random_state=0)
    all_scores["MI"] = mi_scores
    all_ranks["MI_Rank"] = pd.Series(mi_scores).rank(ascending=False, method="min").astype(int)

    # Partial Corr
    print("正在计算 Partial Correlation ...")
    partial_scores = [partial_corr(X_train_fs, y_train_fs, i) for i in range(X_train_fs.shape[1])]
    all_scores["PartialCorr"] = partial_scores
    all_ranks["PartialCorr_Rank"] = pd.Series(np.abs(partial_scores)).rank(ascending=False, method="min").astype(int)


    # ========== 保存当前目标变量的特征选择结果 ==========
    scores_df = pd.DataFrame(all_scores)
    ranks_df = pd.DataFrame(all_ranks)

    output_base_dir = "integrated_outputs" # 统一的输出目录
    # 为每个目标变量创建独立的输出目录
    current_target_output_dir = os.path.join(output_base_dir, current_target_col)
    os.makedirs(current_target_output_dir, exist_ok=True)

    out_scores = os.path.join(current_target_output_dir, f"{current_target_col}_all_scores.csv")
    out_ranks = os.path.join(current_target_output_dir, f"{current_target_col}_all_ranks.csv")

    scores_df.to_csv(out_scores, index=False, encoding="utf-8-sig")
    ranks_df.to_csv(out_ranks, index=False, encoding="utf-8-sig")

    print(f"当前目标变量 {current_target_col} 的所有特征选择方法已完成，结果保存到：\n- {out_scores}\n- {out_ranks}")


    # ========== 调用融合函数 ==========
    print(f"\n======== 正在为目标变量 {current_target_col} 进行特征融合 ========\n")
    fusion_output_dir = os.path.join(output_base_dir, current_target_col, "fused_results")
    os.makedirs(fusion_output_dir, exist_ok=True)

    # 传递 DataFrame 对象而不是文件路径
    scores_out, ranks_out, method_weights, fused_curve, wavelengths = evaluate_methods(
        scores_df, ranks_df, window_size=10, output_dir=fusion_output_dir
    )
    all_fused_results[current_target_col] = {
        "scores_out": scores_out,
        "ranks_out": ranks_out,
        "method_weights": method_weights,
        "fused_curve": fused_curve,
        "wavelengths": wavelengths
    }

print("\n======== 所有目标变量的特征选择和融合已完成 ========\n")
"""

# ===============================================
# 预测模型训练与评估 (原第三段代码的修改)
# ===============================================

print("\n======== 开始预测模型训练与评估 ========\n")

# 所有模型 (与原第三段代码一致，但可以在这里调整)
models_prediction = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01),
    # "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.1),
    "MLP": MLPRegressor(hidden_layer_sizes=(100,50), max_iter=500, random_state=42),
    "CatBoost": CatBoostRegressor(n_estimators=200, random_state=42, verbose=0),
}

# 定义用于模型评估的特征数量 (Top-N)
TOPN_LIST = {
    "Mad": list(range(50, 75, 5)), # 使用较小的范围以加快示例运行速度
    "Vad": list(range(50, 75, 5))
}

# 定义 TopN 列表
# TOPN_LIST = {
#     "Mad": [56, 64, 58, 66, 62, 68, 60, 74, 26, 54, 52, 70, 72, 50, 34],
#     "Vad": [70, 74, 80, 50, 54, 48, 56, 76, 42, 10, 8, 68, 6, 52, 64, 62, 44]
# }

# # 定义 TopN 列表
# TOPN_LIST = {
#     "Mad": [66, 62, 72, 64, 76, 68, 70, 54, 50, 58, 56, 60, 74, 48, 80, 52, 78, 44, 38, 46, 36, 34, 40, 26, 42, 32],
#     "Vad": [54, 60, 72, 64, 62, 58, 68, 78, 6, 80, 56, 76, 74, 70, 66, 48, 50, 42, 52, 10, 38, 8, 44, 12, 40]
# }

def run_experiment_prediction(target, df_train_data, df_test_data, ranks_df, topn_list):
    results = []
    
    # 准备训练集和测试集的 X 和 y
    X_train_full = df_train_data[feature_cols].values
    y_train_full = df_train_data[target].values
    X_test_full = df_test_data[feature_cols].values
    y_test_full = df_test_data[target].values

    methods_to_evaluate = [c for c in ranks_df.columns if c.endswith("_Rank")]

    for method in tqdm(methods_to_evaluate, desc=f"{target} Methods"):
        sorted_wavelengths = ranks_df.sort_values(by=method)["Wavelength"].values

        for model_name, model in tqdm(models_prediction.items(), desc=f"{target} Models ({method})", leave=False):
            row = {"Target": target, "Method": method.replace("_Rank",""), "Model": model_name}
            
            maes = []
            for N in topn_list:
                selected_wavelengths = sorted_wavelengths[:N]
                
                # 检查所有选择的波长是否存在于特征列中
                valid_selected_wavelengths = [w for w in selected_wavelengths if w in df_train_data.columns]
                
                if not valid_selected_wavelengths:
                    print(f"警告: 目标 {target}, 方法 {method}, N={N} 没有找到有效的波长特征。跳过。")
                    row[f"MAE-{N}"] = np.nan
                    continue

                # 从训练集和测试集DataFrame中选择特征
                X_train_selected = df_train_data[valid_selected_wavelengths].values
                X_test_selected = df_test_data[valid_selected_wavelengths].values

                # 在这里，我们不再需要内部的 train_test_split，因为我们已经有了 df_train_data 和 df_test_data
                # 而是直接使用 X_train_selected, y_train_full 进行训练
                # 和 X_test_selected, y_test_full 进行评估

                if X_train_selected.shape[0] == 0 or X_test_selected.shape[0] == 0:
                    print(f"警告: 目标 {target}, 方法 {method}, N={N} 的训练集或测试集为空。跳过。")
                    row[f"MAE-{N}"] = np.nan
                    continue

                model.fit(X_train_selected, y_train_full)
                y_pred = model.predict(X_test_selected)

                mae = mean_absolute_error(y_test_full, y_pred)
                row[f"MAE-{N}"] = mae
                maes.append(mae)

            # 额外加一列：所有 N 上的平均 MAE (忽略 NaN 值)
            row["MAE-mean"] = np.nanmean(maes) if maes else np.nan

            results.append(row)

    return pd.DataFrame(results)

# =======================
# K 折封装：每折运行完整流程（特征选择、融合、预测）
# =======================
def run_pipeline_for_fold(df_train, df_test, fold_idx):
    all_fused_results = {}

    # 特征选择与融合（针对每个目标变量）
    for current_target_col in tqdm(
        target_cols, desc=f"Fold {fold_idx+1} 目标变量", unit="target", dynamic_ncols=True, leave=False
    ):
        print(f"\n======== Fold {fold_idx+1}: 正在处理目标变量 {current_target_col} ========\n")

        # 获取训练集数据
        X_train_fs = df_train[feature_cols].values.astype(float)
        y_train_fs = df_train[current_target_col].values.astype(float)

        # 结果存储
        all_scores = {"Wavelength": feature_cols}
        all_ranks = {"Wavelength": feature_cols}

        # 方法 1: PLS (VIP)
        print("计算 PLS VIP ...")
        scaler_x, scaler_y = StandardScaler(), StandardScaler()
        X_scaled = scaler_x.fit_transform(X_train_fs)
        y_scaled = scaler_y.fit_transform(y_train_fs.reshape(-1, 1)).ravel()

        n_components = min(10, X_train_fs.shape[1], X_train_fs.shape[0] - 1)
        if n_components > 0:
            pls = PLSRegression(n_components=n_components)
            pls.fit(X_scaled, y_scaled)

            t = pls.x_scores_
            w = pls.x_weights_
            q = pls.y_loadings_
            if q.shape[0] > 1:
                q = np.sqrt((q ** 2).sum(axis=0))
            else:
                q = q.ravel()
            p, A = X_train_fs.shape[1], t.shape[1]
            s = np.array([(t[:, a] ** 2).sum() * (q[a] ** 2) for a in range(A)])
            if s.sum() != 0:
                vip = np.sqrt(p * ((w ** 2) @ s) / s.sum())
            else:
                vip = np.zeros(p)

            all_scores["PLS"] = vip
            all_ranks["PLS_Rank"] = pd.Series(vip).rank(ascending=False, method="min").astype(int)
        else:
            print("警告: 数据不足以进行PLS回归，PLS VIP计算跳过。")
            all_scores["PLS"] = np.zeros(len(feature_cols))
            all_ranks["PLS_Rank"] = pd.Series(np.zeros(len(feature_cols))).rank(ascending=False, method="min").astype(int)

        # 方法 2: 树模型 + SHAP
        print("计算 树模型 + SHAP ...")
        X_model_train, X_model_test, y_model_train, y_model_test = train_test_split(
            X_train_fs, y_train_fs, test_size=0.2, random_state=42
        )

        models_fs = {
            "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=200, random_state=42, verbosity=0),
            "LightGBM": LGBMRegressor(n_estimators=200, random_state=42, verbose=-1),
            "CatBoost": CatBoostRegressor(iterations=200, depth=6, learning_rate=0.1,
                                          random_state=42, verbose=0)
        }

        def compute_importance(name, model, X_used, y_train_subset):
            if name in ["RandomForest", "XGBoost", "LightGBM", "CatBoost"]:
                try:
                    if not hasattr(model, "tree_model") and not hasattr(model, "best_iteration_"):
                        model.fit(X_used, y_train_subset)
                    if hasattr(model, 'feature_importances_'):
                        explainer = shap.TreeExplainer(model)
                    else:
                        print(f"警告: 模型 {name} 不支持TreeExplainer或其内部结构不匹配，尝试使用KernelExplainer (较慢)。")
                        if X_used.shape[0] > 100:
                            X_sampling = shap.sample(X_used, 100)
                        else:
                            X_sampling = X_used
                        explainer = shap.KernelExplainer(model.predict, X_sampling)
                    shap_values = explainer.shap_values(X_used)
                    if isinstance(shap_values, list):
                        feature_importance = np.abs(shap_values[0]).mean(axis=0)
                    else:
                        feature_importance = np.abs(shap_values).mean(axis=0)
                except Exception as e:
                    print(f"错误: 计算 {name} 的SHAP值失败: {e}，尝试使用 model.feature_importances_。")
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = model.feature_importances_
                    else:
                        print(f"警告: 模型 {name} 也没有 feature_importances_ 属性，将使用 Permutation Importance。")
                        from sklearn.inspection import permutation_importance
                        result = permutation_importance(model, X_used, y_train_subset, n_repeats=10, random_state=42)
                        feature_importance = result.importances_mean
            else:
                from sklearn.inspection import permutation_importance
                result = permutation_importance(model, X_used, y_train_subset, n_repeats=10, random_state=42)
                feature_importance = result.importances_mean
            return feature_importance

        for name, model in models_fs.items():
            model.fit(X_model_train, y_model_train)
            importance = compute_importance(name, model, X_model_train, y_model_train)
            all_scores[name] = importance
            all_ranks[name + "_Rank"] = pd.Series(importance).rank(ascending=False, method="min").astype(int)

        # 方法 3: 统计方法
        print("计算 统计方法 ...")
        spearman_scores = [spearmanr(X_train_fs[:, i], y_train_fs)[0] if len(np.unique(X_train_fs[:, i])) > 1 and len(np.unique(y_train_fs)) > 1 else 0 for i in range(X_train_fs.shape[1])]
        all_scores["Spearman"] = spearman_scores
        all_ranks["Spearman_Rank"] = pd.Series(np.abs(spearman_scores)).rank(ascending=False, method="min").astype(int)

        print("正在计算 Pearson 相关系数...")
        pearson_scores = [pearsonr(X_train_fs[:, i], y_train_fs)[0] if len(np.unique(X_train_fs[:, i])) > 1 and len(np.unique(y_train_fs)) > 1 else 0 for i in range(X_train_fs.shape[1])]
        all_scores["Pearson"] = pearson_scores
        all_ranks["Pearson_Rank"] = pd.Series(np.abs(pearson_scores)).rank(ascending=False, method="min").astype(int)

        print("正在计算 dCor ...")
        dcor_scores = [dcor.distance_correlation(X_train_fs[:, i], y_train_fs) for i in range(X_train_fs.shape[1])]
        all_scores["dCor"] = dcor_scores
        all_ranks["dCor_Rank"] = pd.Series(dcor_scores).rank(ascending=False, method="min").astype(int)

        print("正在计算 HSIC ...")
        hsic_scores = [hsic(X_train_fs[:, i], y_train_fs) for i in range(X_train_fs.shape[1])]
        all_scores["HSIC"] = hsic_scores
        all_ranks["HSIC_Rank"] = pd.Series(hsic_scores).rank(ascending=False, method="min").astype(int)

        print("正在计算 MI ...")
        mi_scores = mutual_info_regression(X_train_fs, y_train_fs, random_state=0)
        all_scores["MI"] = mi_scores
        all_ranks["MI_Rank"] = pd.Series(mi_scores).rank(ascending=False, method="min").astype(int)

        print("正在计算 Partial Correlation ...")
        partial_scores = [partial_corr(X_train_fs, y_train_fs, i) for i in range(X_train_fs.shape[1])]
        all_scores["PartialCorr"] = partial_scores
        all_ranks["PartialCorr_Rank"] = pd.Series(np.abs(partial_scores)).rank(ascending=False, method="min").astype(int)

        # 保存当前目标变量的特征选择结果（按 Fold 分目录）
        scores_df = pd.DataFrame(all_scores)
        ranks_df = pd.DataFrame(all_ranks)

        current_target_output_dir = os.path.join(output_base_dir, f"Fold_{fold_idx+1}", current_target_col)
        os.makedirs(current_target_output_dir, exist_ok=True)

        out_scores = os.path.join(current_target_output_dir, f"{current_target_col}_all_scores.csv")
        out_ranks = os.path.join(current_target_output_dir, f"{current_target_col}_all_ranks.csv")
        scores_df.to_csv(out_scores, index=False, encoding="utf-8-sig")
        ranks_df.to_csv(out_ranks, index=False, encoding="utf-8-sig")
        print(f"Fold {fold_idx+1} - 目标 {current_target_col} 的特征选择结果保存：\n- {out_scores}\n- {out_ranks}")

        # 融合
        print(f"\n======== Fold {fold_idx+1}: 为目标变量 {current_target_col} 进行特征融合 ========\n")
        fusion_output_dir = os.path.join(output_base_dir, f"Fold_{fold_idx+1}", current_target_col, "fused_results")
        os.makedirs(fusion_output_dir, exist_ok=True)

        scores_out, ranks_out, method_weights, fused_curve, wavelengths = evaluate_methods(
            scores_df, ranks_df, window_size=10, output_dir=fusion_output_dir
        )
        all_fused_results[current_target_col] = {
            "scores_out": scores_out,
            "ranks_out": ranks_out,
            "method_weights": method_weights,
            "fused_curve": fused_curve,
            "wavelengths": wavelengths
        }

    # 预测评估（按 Fold 保存）
    df_mad = run_experiment_prediction(
        "Mad", df_train, df_test, all_fused_results["Mad"]["ranks_out"], TOPN_LIST["Mad"]
    )
    df_mad["Fold"] = fold_idx + 1

    df_vad = run_experiment_prediction(
        "Vad", df_train, df_test, all_fused_results["Vad"]["ranks_out"], TOPN_LIST["Vad"]
    )
    df_vad["Fold"] = fold_idx + 1

    prediction_output_dir = os.path.join(output_base_dir, f"Fold_{fold_idx+1}", "prediction_results")
    os.makedirs(prediction_output_dir, exist_ok=True)

    df_mad.to_csv(os.path.join(prediction_output_dir, "mad_prediction_results.csv"), index=False)
    df_vad.to_csv(os.path.join(prediction_output_dir, "vad_prediction_results.csv"), index=False)
    print(f"Fold {fold_idx+1} 预测结果已保存到：\n- {os.path.join(prediction_output_dir, 'mad_prediction_results.csv')}\n- {os.path.join(prediction_output_dir, 'vad_prediction_results.csv')}")

    return df_mad, df_vad


# =======================
# 运行 K 折交叉验证并汇总结果
# =======================
n_splits = 5  # 可调整折数
kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)

all_mad_results = []
all_vad_results = []

for fold_idx, (train_index, test_index) in tqdm(
    enumerate(kf.split(df_original)), total=n_splits, desc="KFold CV", unit="fold", dynamic_ncols=True
):
    print(f"\n================ 开始 Fold {fold_idx+1}/{n_splits} ================")
    df_train = df_original.iloc[train_index].copy()
    df_test = df_original.iloc[test_index].copy()
    print(f"Fold {fold_idx+1} - 训练集: {df_train.shape}, 测试集: {df_test.shape}")

    df_mad_fold, df_vad_fold = run_pipeline_for_fold(df_train, df_test, fold_idx)
    all_mad_results.append(df_mad_fold)
    all_vad_results.append(df_vad_fold)

# 汇总与平均
mad_all = pd.concat(all_mad_results, ignore_index=True)
vad_all = pd.concat(all_vad_results, ignore_index=True)

agg_dir = os.path.join(output_base_dir, "prediction_results_kfold")
os.makedirs(agg_dir, exist_ok=True)

mad_all.to_csv(os.path.join(agg_dir, "mad_prediction_results_all_folds.csv"), index=False)
vad_all.to_csv(os.path.join(agg_dir, "vad_prediction_results_all_folds.csv"), index=False)

mad_avg = mad_all.groupby(["Target", "Method", "Model"]).mean(numeric_only=True).reset_index()
vad_avg = vad_all.groupby(["Target", "Method", "Model"]).mean(numeric_only=True).reset_index()

mad_avg.to_csv(os.path.join(agg_dir, "mad_prediction_results_mean.csv"), index=False)
vad_avg.to_csv(os.path.join(agg_dir, "vad_prediction_results_mean.csv"), index=False)

print(f"\nK 折交叉验证已完成。汇总结果已保存到：\n- {os.path.join(agg_dir, 'mad_prediction_results_all_folds.csv')}\n- {os.path.join(agg_dir, 'vad_prediction_results_all_folds.csv')}\n- {os.path.join(agg_dir, 'mad_prediction_results_mean.csv')}\n- {os.path.join(agg_dir, 'vad_prediction_results_mean.csv')}")

# =======================
# 聚合各折的特征选择 scores 和 ranks（取均值）以及融合结果
# =======================
print("\n======== 开始聚合各折的特征选择 scores/ranks 与融合结果（均值） ========\n")

def aggregate_feature_selection_and_fused(output_base_dir, target):
    fold_dirs = sorted([d for d in os.listdir(output_base_dir) if d.startswith("Fold_")])
    score_dfs, rank_dfs = [], []
    fused_score_dfs, fused_rank_dfs = [], []

    for fd in fold_dirs:
        base = os.path.join(output_base_dir, fd, target)
        try:
            df_scores = pd.read_csv(os.path.join(base, f"{target}_all_scores.csv")).set_index("Wavelength")
            score_dfs.append(df_scores)
        except Exception as e:
            print(f"提示: 跳过 {fd}/{target} scores，原因: {e}")
        try:
            df_ranks = pd.read_csv(os.path.join(base, f"{target}_all_ranks.csv")).set_index("Wavelength")
            rank_dfs.append(df_ranks)
        except Exception as e:
            print(f"提示: 跳过 {fd}/{target} ranks，原因: {e}")

        fused_dir = os.path.join(base, "fused_results")
        try:
            df_fused_scores = pd.read_csv(os.path.join(fused_dir, "scores_with_fused.csv")).set_index("Wavelength")
            fused_score_dfs.append(df_fused_scores)
        except Exception as e:
            print(f"提示: 跳过 {fd}/{target} fused scores，原因: {e}")
        try:
            df_fused_ranks = pd.read_csv(os.path.join(fused_dir, "ranks_with_fused.csv")).set_index("Wavelength")
            fused_rank_dfs.append(df_fused_ranks)
        except Exception as e:
            print(f"提示: 跳过 {fd}/{target} fused ranks，原因: {e}")

    # 输出目录（顶层）
    out_target_dir = os.path.join(output_base_dir, target)
    os.makedirs(out_target_dir, exist_ok=True)
    out_fused_dir = os.path.join(out_target_dir, "fused_results")
    os.makedirs(out_fused_dir, exist_ok=True)

    # 聚合分数（后续用于派生排名）
    avg_scores = None
    if score_dfs:
        avg_scores = pd.concat(score_dfs).groupby(level=0).mean()
        avg_scores.reset_index().to_csv(
            os.path.join(out_target_dir, f"{target}_all_scores.csv"),
            index=False, encoding="utf-8-sig"
        )
        print(f"✅ 已保存 {target} 的平均分数到: {os.path.join(out_target_dir, f'{target}_all_scores.csv')}")
    else:
        print(f"⚠️ 未找到 {target} 的 scores 文件，跳过分数聚合。")

    # 聚合排名（基于平均分数重新计算名次，不做rank均值）
    if avg_scores is not None:
        derived_ranks = pd.DataFrame(index=avg_scores.index)
        for col in avg_scores.columns:
            rank_col = f"{col}_Rank" if not col.endswith("_Rank") else col
            derived_ranks[rank_col] = avg_scores[col].rank(ascending=False, method="min").astype(int)
        derived_ranks.reset_index().to_csv(
            os.path.join(out_target_dir, f"{target}_all_ranks.csv"),
            index=False, encoding="utf-8-sig"
        )
        print(f"✅ 已保存 {target} 的派生排名到: {os.path.join(out_target_dir, f'{target}_all_ranks.csv')}（基于平均分数排序得到的整数名次）")
    else:
        print(f"⚠️ 未找到 {target} 的平均分数，无法根据分数派生排名。")

    # 聚合融合分数（后续用于派生融合排名）
    avg_fused_scores = None
    if fused_score_dfs:
        avg_fused_scores = pd.concat(fused_score_dfs).groupby(level=0).mean()
        avg_fused_scores.reset_index().to_csv(
            os.path.join(out_fused_dir, "scores_with_fused.csv"),
            index=False, encoding="utf-8-sig"
        )
        print(f"✅ 已保存 {target} 的平均融合分数到: {os.path.join(out_fused_dir, 'scores_with_fused.csv')}")
    else:
        print(f"⚠️ 未找到 {target} 的 fused scores 文件，跳过融合分数聚合。")

    # 聚合融合排名（基于平均融合分数重新计算名次，不做rank均值）
    if avg_fused_scores is not None:
        derived_fused_ranks = pd.DataFrame(index=avg_fused_scores.index)
        for col in avg_fused_scores.columns:
            # Fused_Score 专用列名，其它方法加 _Rank 后缀以保持一致
            if col == "Fused_Score":
                derived_fused_ranks["Fused_Rank"] = avg_fused_scores[col].rank(ascending=False, method="min").astype(int)
            else:
                derived_fused_ranks[f"{col}_Rank"] = avg_fused_scores[col].rank(ascending=False, method="min").astype(int)
        derived_fused_ranks.reset_index().to_csv(
            os.path.join(out_fused_dir, "ranks_with_fused.csv"),
            index=False, encoding="utf-8-sig"
        )
        print(f"✅ 已保存 {target} 的派生融合排名到: {os.path.join(out_fused_dir, 'ranks_with_fused.csv')}（基于平均融合分数排序得到的整数名次）")
    else:
        print(f"⚠️ 未找到 {target} 的平均融合分数，无法根据分数派生融合排名。")


# 对两个目标分别聚合
for t in target_cols:
    aggregate_feature_selection_and_fused(output_base_dir, t)

print("\n======== 聚合完成：后续绘图将统一使用均值文件 ========\n")



# =======================
# 方法在不同模型上的表现可视化（heatmap/bar/box/radar）
# =======================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_methods(pivot_df, plot_type="heatmap"):
    """
    可视化方法在不同模型上的表现
    pivot_df: 行是 (Model, Target)，列是 Method
    plot_type: "heatmap", "bar", "box", "radar"
    """

    # 把透视表变成长表
    df_reset = pivot_df.reset_index().melt(
        id_vars=["Model", "Target"],
        var_name="Method",
        value_name="MAE-mean"
    )

    if plot_type == "heatmap":
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            pivot_df.T,  # 转置，横轴是Model，纵轴是Method
            annot=True, fmt="d", cmap="YlGnBu",
            cbar_kws={'label': 'MAE-mean'}
        )
        plt.title("Performance Heatmap (Method × Model+Target)")
        plt.xlabel("Model, Target")
        plt.ylabel("Method")
        plt.tight_layout()
        plt.show()

    elif plot_type == "bar":
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=df_reset,
            x="Model", y="MAE-mean", hue="Method"
        )
        plt.title("Comparison of Methods across Models")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    elif plot_type == "box":
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_reset, x="Model", y="MAE-mean", hue="Method")
        plt.title("Distribution of MAE across Models for each Method")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    elif plot_type == "radar":
        methods = pivot_df.columns
        models = pivot_df.index

        angles = np.linspace(0, 2 * np.pi, len(models), endpoint=False).tolist()
        angles += angles[:1]  # 闭合角度

        plt.figure(figsize=(7, 7))
        ax = plt.subplot(111, polar=True)

        for method in methods:
            values = pivot_df[method].values
            values = np.concatenate((values, [values[0]]))  # 闭合
            ax.plot(angles, values, "o-", linewidth=2, label=method)
            ax.fill(angles, values, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([f"{m}-{t}" for m, t in models], rotation=30)
        plt.title("Radar Plot of Methods across Models")
        plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.05))
        plt.tight_layout()
        plt.show()

    else:
        raise ValueError("plot_type 必须是 'heatmap', 'bar', 'box', 'radar' 之一")

def rescale_per_model(pivot_df, lower=0.2, upper=1.0):
    """
    针对每个 Model-Target 行做 min-max 归一化，并设置下限 > 0
    """
    scaled = pivot_df.copy()
    for idx in scaled.index:
        row = scaled.loc[idx]
        rmin, rmax = row.min(), row.max()
        if rmax > rmin:
            scaled.loc[idx] = lower + (row - rmin) / (rmax - rmin) * (upper - lower)
        else:
            scaled.loc[idx] = (lower + upper) / 2  # 如果所有值一样
    return scaled

# 从 K 折平均结果构造透视表
combined_avg = pd.concat([mad_avg, vad_avg], ignore_index=True)
pivot_df = combined_avg.pivot_table(index=["Model", "Target"], columns="Method", values="MAE-mean")

# 画排序热力图：越小越好，使用排名整数便于直观比较
pivot_rank = pivot_df.rank(axis=1, method="min", ascending=True).astype(int)
visualize_methods(pivot_rank, plot_type="heatmap")

# 根据指定顺序对方法列进行排序，并补充箱线图与归一化雷达图
desired_order = [
    'Fused', 'CatBoost', 'HSIC', 'LightGBM', 'MI', 'PLS',
    'PartialCorr', 'Pearson', 'RandomForest', 'Spearman', 'XGBoost', 'dCor'
]

# 过滤掉不存在的列，避免错误
existing_methods_in_order = [m for m in desired_order if m in pivot_rank.columns]
if len(existing_methods_in_order) == 0:
    existing_methods_in_order = list(pivot_rank.columns)

# 应用新的列顺序
pivot_rank_sorted = pivot_rank[existing_methods_in_order]
print("\n--- 使用指定顺序生成排序后的热力图 ---")
visualize_methods(pivot_rank_sorted, plot_type="heatmap")

# 按方法的平均排名排序后绘制箱线图（使用原始 MAE 值）
print("\n--- 生成箱线图 (按平均排名排序) ---")
method_avg_rank = pivot_rank_sorted.mean().sort_values()
box_df = pivot_df[method_avg_rank.index]
visualize_methods(box_df, plot_type="box")

# 生成归一化后的雷达图（先按每个 Model-Target 行做 min-max 归一化）
print("\n--- 生成雷达图（按每行归一化） ---")
radar_df = rescale_per_model(box_df, lower=0.2, upper=1.0)
visualize_methods(radar_df, plot_type="radar")

# ===============================================
# 3. 结果可视化 (新增部分)
# ===============================================
print("\n======== 开始生成可视化图表 ========\n")


# --- 图 1: 所有独立特征选择方法的热力图 ---
def plot_individual_method_heatmaps(base_dir, save_path):
    """
    绘制并保存Mad和Vad在所有独立方法下的重要性分数热力图。
    """
    try:
        # 读取两个目标变量的分数数据
        mad_scores_path = os.path.join(base_dir, "Mad", "Mad_all_scores.csv")
        vad_scores_path = os.path.join(base_dir, "Vad", "Vad_all_scores.csv")

        mad_df = pd.read_csv(mad_scores_path).set_index("Wavelength")
        vad_df = pd.read_csv(vad_scores_path).set_index("Wavelength")

        # 对每个方法分别进行min-max归一化以便于可视化
        mad_df_norm = (mad_df - mad_df.min()) / (mad_df.max() - mad_df.min())
        vad_df_norm = (vad_df - vad_df.min()) / (vad_df.max() - vad_df.min())

        # 创建图形
        fig, axes = plt.subplots(2, 1, figsize=(16, 14), sharex=True)

        # 绘制MAD归一化分数热力图
        sns.heatmap(mad_df_norm.T, cmap="viridis", ax=axes[0], cbar_kws={'label': 'Normalized Score'})
        axes[0].set_title("Heatmap of Individual Methods across Wavelengths (Mad)", fontsize=14)
        axes[0].set_ylabel("Method")

        # 绘制VAD归一化分数热力图
        sns.heatmap(vad_df_norm.T, cmap="viridis", ax=axes[1], cbar_kws={'label': 'Normalized Score'})
        axes[1].set_title("Heatmap of Individual Methods across Wavelengths (Vad)", fontsize=14)
        axes[1].set_ylabel("Method")

        plt.xlabel("Wavelength")
        plt.tight_layout()
        
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format="svg", dpi=300)
        plt.show()
        print(f"✅ 所有独立方法的热力图已保存到: {save_path}")

    except FileNotFoundError as e:
        print(f"错误: 找不到文件 {e.filename}。请确保前面的步骤已成功运行并生成了对应的 .csv 文件。")
    except Exception as e:
        print(f"绘制独立方法热力图时发生错误: {e}")


# --- 图 2: 融合后结果的热力图 ---
def plot_fused_results_heatmap(base_dir, save_path):
    """
    绘制并保存Mad和Vad融合后的重要性分数的热力图。
    """
    try:
        # 读取两个融合后的得分文件
        dir_mad = os.path.join(base_dir, "Mad", "fused_results")
        dir_vad = os.path.join(base_dir, "Vad", "fused_results")
        
        df_mad = pd.read_csv(os.path.join(dir_mad, "scores_with_fused.csv"))
        df_vad = pd.read_csv(os.path.join(dir_vad, "scores_with_fused.csv"))

        wavelengths_mad = df_mad["Wavelength"].values
        fused_curve_mad = df_mad["Fused_Score"].values

        wavelengths_vad = df_vad["Wavelength"].values
        fused_curve_vad = df_vad["Fused_Score"].values

        # 创建画布和子图
        fig, axes = plt.subplots(2, 1, figsize=(14, 4), sharex=True)

        # Mad 热力图（放上面）
        im1 = axes[0].imshow(
            fused_curve_mad.reshape(1, -1), aspect="auto", cmap="viridis",
            extent=[wavelengths_mad.min(), wavelengths_mad.max(), 0, 1]
        )
        axes[0].set_yticks([])
        axes[0].set_ylabel("Mad", rotation=0, labelpad=20, va='center')
        axes[0].set_title("Fused Importance Heatmap")

        # Vad 热力图（放下面）
        im2 = axes[1].imshow(
            fused_curve_vad.reshape(1, -1), aspect="auto", cmap="viridis",
            extent=[wavelengths_vad.min(), wavelengths_vad.max(), 0, 1]
        )
        axes[1].set_yticks([])
        axes[1].set_ylabel("Vad", rotation=0, labelpad=20, va='center')
        axes[1].set_xlabel("Wavelength")

        # 自动排版，为右侧的 colorbar 留出空间
        plt.tight_layout(rect=[0, 0, 0.92, 1])

        # 在右侧单独添加一个共享的 colorbar 轴
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.76])
        fig.colorbar(im2, cax=cbar_ax, label="Fused Importance Score")
        
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format="svg", dpi=300)
        plt.show()
        print(f"✅ 融合结果热力图已保存到: {save_path}")

    except FileNotFoundError as e:
        print(f"错误: 找不到文件 {e.filename}。请确保融合步骤已成功运行并生成了 'scores_with_fused.csv' 文件。")
    except Exception as e:
        print(f"绘制融合结果热力图时发生错误: {e}")


# --- 调用绘图函数 ---
# 定义图片保存的目录
image_output_dir = os.path.join(output_base_dir, "visualization_images")

# 绘制并保存第一张图
plot_individual_method_heatmaps(
    base_dir=output_base_dir,
    save_path=os.path.join(image_output_dir, "individual_methods_heatmap.svg")
)

# 绘制并保存第二张图
plot_fused_results_heatmap(
    base_dir=output_base_dir,
    save_path=os.path.join(image_output_dir, "fused_results_heatmap.svg")
)

print("\n======== 可视化图表生成完毕 ========\n")

# =========================
# 仅绘制 M_ad & V_ad 堆叠柱状图（左右子图）
# =========================
def plot_stacked_bars(mad_file, vad_file, window_size=10, save_path="stacked_results.svg", weighted=False):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)  # 横向两个子图

    # Target 显示映射
    target_labels = {
        "Mad": r"$M_{ad}$",
        "Vad": r"$V_{ad}$",
    }

    def load_scores_and_ranks(scores_path):
        # 读取 scores.csv，如果不存在则抛错
        df_scores = pd.read_csv(scores_path)
        if "Wavelength" not in df_scores.columns:
            raise ValueError(f"文件 {scores_path} 缺少 'Wavelength' 列")
        methods = [c for c in df_scores.columns if c != "Wavelength"]
        # 派生 ranks（每个方法列按分数降序排名）
        ranks_dict = {"Wavelength": df_scores["Wavelength"].values}
        for m in methods:
            ranks_dict[f"{m}_Rank"] = pd.Series(df_scores[m]).rank(ascending=False, method="min").astype(int)
        df_ranks = pd.DataFrame(ranks_dict)
        return df_scores, df_ranks

    for ax, file, title in zip(
        axes,
        [mad_file, vad_file],
        ["Mad", "Vad"]
    ):
        try:
            df_scores, df_ranks = load_scores_and_ranks(file)
            # 评估融合指标（使用现有 evaluate_methods 接口）
            out_dir = os.path.join(output_base_dir, title, "fused_results_for_stacked")
            scores_out, ranks_out, results_df, fused_curve, wavelengths = evaluate_methods(
                df_scores, df_ranks, window_size=window_size, output_dir=out_dir
            )

            # 重命名列（NMF_Weight -> Consistency）
            results_df = results_df.rename(columns={"NMF_Weight": "Consistency"})
            # 根据 weighted 标志决定是否对三个指标进行加权
            if weighted:
                if {"Weight_Consistency", "Weight_Smoothness", "Weight_Segmentation"}.issubset(results_df.columns):
                    results_df["Consistency"] = results_df["Consistency"] * results_df["Weight_Consistency"]
                    results_df["Smoothness"] = results_df["Smoothness"] * results_df["Weight_Smoothness"]
                    results_df["Segmentation"] = results_df["Segmentation"] * results_df["Weight_Segmentation"]
                else:
                    print("提示：未检测到动态权重列，堆叠图将退化为未加权显示。")
            
            metrics = ["Consistency", "Smoothness", "Segmentation"]
            # 按 Final_Weight 排序（方法层面）
            plot_df = results_df.sort_values("Final_Weight", ascending=False)[metrics]

            # 堆叠柱状图
            plot_df.plot(
                kind="bar",
                stacked=True,
                ax=ax,
                colormap="tab20c",
                width=0.8,
                edgecolor="black"
            )

            ax.set_ylabel("Score Contribution")
            if weighted:
                ax.set_title(f"Weighted Stacked Bar of Method Scores ({target_labels[title]})")
            else:
                ax.set_title(f"Stacked Bar of Method Scores ({target_labels[title]})")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.legend(title="Metrics")
        except FileNotFoundError as e:
            ax.set_title(f"文件缺失: {e.filename}")
            ax.text(0.5, 0.5, f"缺少文件\n{e.filename}", ha="center", va="center")
        except Exception as e:
            ax.set_title("绘图错误")
            ax.text(0.5, 0.5, f"错误: {e}", ha="center", va="center")

    plt.tight_layout()
    # 保存为 SVG
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, format="svg")
    plt.show()


# 使用示例（未加权堆叠）
plot_stacked_bars(
    os.path.join(output_base_dir, "Mad", "Mad_all_scores.csv"),
    os.path.join(output_base_dir, "Vad", "Vad_all_scores.csv"),
    save_path=os.path.join(output_base_dir, "visualization_images", "stacked_results_unweighted.svg"),
    weighted=False
)

# 使用示例（加权堆叠）
plot_stacked_bars(
    os.path.join(output_base_dir, "Mad", "Mad_all_scores.csv"),
    os.path.join(output_base_dir, "Vad", "Vad_all_scores.csv"),
    save_path=os.path.join(output_base_dir, "visualization_images", "stacked_results_weighted.svg"),
    weighted=True
)