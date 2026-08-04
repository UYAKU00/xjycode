import os
import sys
import torch
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr


# 强制使用交互式后端
try:
    matplotlib.use('TkAgg')
except:
    pass

# ================= 1. 配置 =================
PROJECT_ROOT = r"D:\Digital_twin\step_two"
TRAIN_CODE_DIR = os.path.join(PROJECT_ROOT, "train")

if TRAIN_CODE_DIR not in sys.path:
    sys.path.insert(0, TRAIN_CODE_DIR)

from dataset_loader import PhysioDataset
from model_def import TwinFusionNet

MOD_ROOT = os.path.join(PROJECT_ROOT, "mod_1_11_no")
STRATEGY_PATH = os.path.join(PROJECT_ROOT, r"data\processed_all\cv_strategy.npy")

PARAM_NAMES = [
    'R_m', 'R_a', 'R_haa', 'R_lna', 'R_lca', 'R_aop', 'R_rula', 'R_rica', 'R_lica', 'R_lula',
    'R_rsv', 'R_rijv', 'R_lijv', 'R_lsv', 'R_sv', 'R_t', 'R_p', 'R_rpap', 'R_lpap', 'R_rpad',
    'R_lpad', 'R_rpv', 'R_lpv',
    'C_haa', 'C_lna', 'C_lca', 'C_aop', 'C_rula', 'C_rica', 'C_lica', 'C_lula', 'C_sap',
    'C_rsv', 'C_rijv', 'C_lijv', 'C_lsv', 'C_sv', 'C_vc', 'C_rpap', 'C_lpap', 'C_rpad',
    'C_lpad', 'C_rpv', 'C_lpv'
]


def run_cross_val_scatter():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载元数据
    meta = np.load(STRATEGY_PATH, allow_pickle=True).item()
    norm_meta = meta['norm_meta']
    y_log_min = torch.tensor(norm_meta['y_log_min'], dtype=torch.float32).to(DEVICE)
    y_log_max = torch.tensor(norm_meta['y_log_max'], dtype=torch.float32).to(DEVICE)

    with open(os.path.join(MOD_ROOT, "test_registry.json"), 'r') as f:
        test_registry = json.load(f)

    all_preds_phys = []
    all_trues_phys = []

    # 2. 提取数据 (示例仅跑第一折，如需全部请改为 range(1, 11))
    print("🚀 正在集结测试集数据...")
    for t_idx in range(1, 11):
        fold_key = f"train_{t_idx}"
        model_path = os.path.join(MOD_ROOT, fold_key, "val_model.pth")
        if not os.path.exists(model_path): continue

        model = TwinFusionNet(n_channels=11, n_features=2, n_outputs=44).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        model.eval()

        test_ds = PhysioDataset(test_registry[fold_key], norm_meta, is_train=False)
        loader = DataLoader(test_ds, batch_size=128, shuffle=False)

        with torch.no_grad():
            for w, v, y, _ in loader:
                out_norm = model(w.to(DEVICE), v.to(DEVICE))
                p_log = out_norm * (y_log_max - y_log_min + 1e-8) + y_log_min
                t_log = y.to(DEVICE) * (y_log_max - y_log_min + 1e-8) + y_log_min
                all_preds_phys.append(torch.exp(p_log).cpu().numpy())
                all_trues_phys.append(torch.exp(t_log).cpu().numpy())
        print(f"   ✅ Fold {t_idx} 数据提取完成")

    # 3. 数据转换 (必须在函数内)
    preds = np.vstack(all_preds_phys)
    trues = np.vstack(all_trues_phys)

    # 逐列归一化用于散点图对比
    scaler = MinMaxScaler()
    trues_norm = scaler.fit_transform(trues)
    preds_norm = scaler.transform(preds)



    # ==========命令行打印 R2 指标 ========
    # 计算所有参数的 R2
    r2_pearson_results = []
    for i in range(len(PARAM_NAMES)):
        t_col = trues_norm[:, i]
        p_col = preds_norm[:, i]
        r2 = r2_score(t_col, p_col)
        try:
            corr, _ = pearsonr(t_col, p_col)
        except:
            corr = 0.0
        r2_pearson_results.append({
            'name': PARAM_NAMES[i],
            'r2': r2,
            'corr': corr
        })

    # 分组
    r_group = [item for item in r2_pearson_results if item['name'].startswith('R_')]
    c_group = [item for item in r2_pearson_results if item['name'].startswith('C_')]

    # 分别按 R2 从高到低排序
    r_group.sort(key=lambda x: x['r2'], reverse=True)
    c_group.sort(key=lambda x: x['r2'], reverse=True)

    print("\n" + "=" * 95)
    # 表头设计：增加 r 这一列
    header = f"{'R参数':<10} : {'R²':>7} | {'r':>6}     ||     {'C参数':<10} : {'R²':>7} | {'r':>6}"
    print(header)
    print("-" * 95)

    # 循环打印（取两者中较长的长度，防止索引溢出）
    max_len = max(len(r_group), len(c_group))
    for i in range(max_len):
        # 处理 R 列
        if i < len(r_group):
            d = r_group[i]
            r_str = f"{d['name']:<10} : {d['r2']:>7.4f} | {d['corr']:>6.4f}"
        else:
            r_str = f"{'':<10}   {'':>7}   {'':>6}"

        # 处理 C 列
        if i < len(c_group):
            d = c_group[i]
            c_str = f"{d['name']:<10} : {d['r2']:>7.4f} | {d['corr']:>6.4f}"
        else:
            c_str = f"{'':<10}   {'':>7}   {'':>6}"

        print(f"{r_str}       |       {c_str}")

    # 打印总体平均值
    avg_r2 = np.mean([item['r2'] for item in r2_pearson_results])
    print("-" * 65)
    print(f"所有 44 个参数的总平均 R²: {avg_r2:.4f}")
    print("=" * 65 + "\n")



    # ================= 4. 分参数分页散点图 (每页12个) =================
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    num_params = len(PARAM_NAMES)
    params_per_page = 12
    pages = (num_params + params_per_page - 1) // params_per_page

    for page in range(pages):
        fig, axes = plt.subplots(3, 4, figsize=(18, 10))
        axes = axes.flatten()
        start_idx = page * params_per_page
        end_idx = min(start_idx + params_per_page, num_params)

        for i, param_idx in enumerate(range(start_idx, end_idx)):
            ax = axes[i]
            t_col = trues_norm[:, param_idx]
            p_col = preds_norm[:, param_idx]

            r2 = r2_score(t_col, p_col)
            ax.scatter(t_col, p_col, s=3, c='#1f77b4', alpha=0.3)
            ax.plot([0, 1], [0, 1], 'r--', lw=1.5)

            ax.set_xlim(0, 1);
            ax.set_ylim(0, 1)
            ax.set_title(f"{PARAM_NAMES[param_idx]}\n$R^2$={r2:.3f}", fontsize=15)
            ax.grid(True, linestyle=':', alpha=0.5)

        # 删除该页多余的子图
        for k in range(end_idx - start_idx, len(axes)):
            fig.delaxes(axes[k])

        fig.suptitle(f"44参数详细回归分析 (第 {page + 1}/{pages} 页)", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # ========================================================================



    # ================= 5. MAPE 误差排名图 =================
    plt.figure(num="误差分析排名", figsize=(10, 8))
    # 计算实际物理量纲下的 MAPE
    mapes = np.mean(np.abs((trues - preds) / (trues + 1e-10)), axis=0) * 100
    sorted_idx = np.argsort(mapes)

    plt.barh(np.arange(num_params), mapes[sorted_idx], color='skyblue')
    plt.yticks(np.arange(num_params), [PARAM_NAMES[i] for i in sorted_idx], fontsize=8)
    plt.gca().invert_yaxis()
    plt.xlabel("MAPE (%)")
    plt.title("44个生理参数预测误差(MAPE)排名", fontsize=13)
    plt.axvline(x=10, color='red', linestyle='--', alpha=0.5, label='10%误差线')
    plt.grid(axis='x', linestyle=':', alpha=0.7)
    plt.tight_layout()

    # ===================== 🌟 组会参数分析 🌟 ==============================
    # 打印结果
    comb_groups = {
        "肺循环总阻力 (R_pulm_total)": [16, 17, 18, 19, 20],  # Rp, Rrpap, Rlpap, Rrpad, Rlpad
        "静脉系统总顺应性 (C_ven_total)": [32, 33, 34, 35, 36],  # Crsv, Crijv, Clijv, Csv, Cvc
        "体循环远端总阻力 (R_dist_total)": [1, 8, 12],  # Ra, Rlica, Rlijv
        "肺部总顺应性 (C_pulm_total)": [37, 38, 39, 40]  # Crpap, Clpap, Crpad, Clpad
    }

    print("\n" + "⭐" * 5 + " 组合等效参数反演指标统计 " + "⭐" * 5)
    print(f"{'组合物理量名称':<30} | {'R² (决定系数)':>12} | {'r (相关系数)':>12} | {'MAPE (%)':>10}")
    print("-" * 85)

    for g_name, idxs in comb_groups.items():
        # 核心：在真实物理量纲上执行求和（电阻串联，电容并联）
        g_true = np.sum(trues[:, idxs], axis=1)
        g_pred = np.sum(preds[:, idxs], axis=1)

        # 计算统计指标
        # R2 决定系数
        g_r2 = r2_score(g_true, g_pred)

        # 皮尔逊相关系数 r
        try:
            g_corr, _ = pearsonr(g_true, g_pred)
        except:
            g_corr = 0.0

        # 平均绝对百分比误差 MAPE
        g_mape = np.mean(np.abs((g_true - g_pred) / (g_true + 1e-10))) * 100

        # 打印这一行的结果
        print(f"{g_name:<30} | {g_r2:>12.4f} | {g_corr:>12.4f} | {g_mape:>10.2f}%")

    print("=" * 85 + "\n")

    # ------------- 窗口 3: 组合参数回归散点图 ----------------
    fig_comb, axes_comb = plt.subplots(2, 2, figsize=(12, 10))
    axes_comb = axes_comb.flatten()
    fig_comb.canvas.manager.set_window_title("组合等效参数验证")

    for i, (g_name, idxs) in enumerate(comb_groups.items()):
        ax = axes_comb[i]
        g_true = np.sum(trues[:, idxs], axis=1)
        g_pred = np.sum(preds[:, idxs], axis=1)

        # 为了绘图美观，对组合量也做一次局部归一化显示
        g_scaler = MinMaxScaler()
        t_plot = g_scaler.fit_transform(g_true.reshape(-1, 1)).flatten()
        p_plot = g_scaler.transform(g_pred.reshape(-1, 1)).flatten()

        r2 = r2_score(g_true, g_pred)
        ax.scatter(t_plot, p_plot, s=10, c='darkorange', alpha=0.4)
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_title(f"{g_name}\n$R^2$: {r2:.4f}", fontsize=20)
        ax.set_xlabel("真实值 (归一化)", fontsize=17)
        ax.set_ylabel("预测值 (归一化)", fontsize=17)
        ax.tick_params(axis='both', labelsize=15)
        ax.grid(True, linestyle=':')
    plt.tight_layout()


    print("✨ 分析图表已全部生成")
    plt.show()



    # ===================== 🌟 11通道结果保存 (彻底消除警告) 🌟 ==============================
    # 1. 确定保存路径
    SAVE_DIR = r"D:\Digital_twin\step_three\all_data_no"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    final_save_path = os.path.join(SAVE_DIR, "data_11_no.csv")

    # 2. 构造单体参数表
    df_single = pd.DataFrame(r2_pearson_results)
    # 计算 MAPE (物理量纲下)
    single_mapes = np.mean(np.abs((trues - preds) / (trues + 1e-10)), axis=0) * 100
    df_single['mape'] = single_mapes
    df_single.insert(0, '数据类别', '单体参数')

    # 3. 构造组合参数表
    comb_data_list = []
    for g_name, idxs in comb_groups.items():
        g_true = np.sum(trues[:, idxs], axis=1)
        g_pred = np.sum(preds[:, idxs], axis=1)
        g_r2 = r2_score(g_true, g_pred)
        try:
            g_corr, _ = pearsonr(g_true, g_pred)
        except:
            g_corr = 0.0
        g_mape = np.mean(np.abs((g_true - g_pred) / (g_true + 1e-10))) * 100
        comb_data_list.append({
            '数据类别': '组合参数',
            'name': g_name,
            'r2': g_r2,
            'corr': g_corr,
            'mape': g_mape
        })
    df_comb = pd.DataFrame(comb_data_list)

    # 4. 构造总体汇总表 (关键：显式填充 0.0 以消除 NA 警告)
    df_avg = pd.DataFrame([{
        '数据类别': '总体统计',
        'name': '所有 44 个参数总平均 R2',
        'r2': avg_r2,
        'corr': 0.0,  # 显式补齐
        'mape': 0.0  # 显式补齐
    }])

    # 5. 统一列顺序并合并
    cols = ['数据类别', 'name', 'r2', 'corr', 'mape']
    # 确保三个 DataFrame 都有相同的列，且没有空列参与合并
    df_all_stats = pd.concat([df_single[cols], df_comb[cols], df_avg[cols]], ignore_index=True)

    # 6. 写入文件
    df_all_stats.to_csv(final_save_path, index=False, encoding='utf-8-sig')

    print(f"💾 11通道统计数据已保存至: {final_save_path}")



if __name__ == "__main__":
    run_cross_val_scatter()


'''
    # B. 拉平所有参数的数据点
    all_t = trues_norm.flatten()
    all_p = preds_norm.flatten()


    # --- 窗口 1: 散点图 ---
    plt.figure(num="整体拟合散点图", figsize=(9, 8))

    # s=0.5 为点的大小，alpha=0.2 为透明度
    plt.scatter(all_t, all_p, s=0.5, c='#1f77b4', alpha=0.2)

    # 绘制红色理想线 (y=x)
    plt.plot([0, 1], [0, 1], 'r--', lw=2)

    # 强制设置范围为 0 到 1
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # 计算全局决定系数
    r2 = np.corrcoef(all_t, all_p)[0, 1] ** 2
    plt.title(f"全参数整体拟合散点图 (逐列归一化)\nGlobal R2 = {r2:.4f}", fontsize=13)
    plt.xlabel("真实值 (归一化)")
    plt.ylabel("预测值 (归一化)")
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
'''

