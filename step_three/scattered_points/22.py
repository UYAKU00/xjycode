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
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler  # 用于统一量纲

#-----70循环几次---------


# 强制使用交互式后端，确保窗口能弹出
try:
    matplotlib.use('TkAgg')
except:
    pass

# ================= 1. 路径与环境强制配置 =================
PROJECT_ROOT = r"D:\Digital_twin\step_two_2"
TRAIN_CODE_DIR = os.path.join(PROJECT_ROOT, "train")

if TRAIN_CODE_DIR not in sys.path:
    sys.path.insert(0, TRAIN_CODE_DIR)

try:
    from dataset import PhysioDataset
    from model import TwinFusionNet

    print("✅ 成功加载模型与数据集定义")
except ImportError:
    print(f"❌ 还是找不到 dataset_loader.py！请确认它是否在: {TRAIN_CODE_DIR}")
    sys.exit()

MOD_ROOT = os.path.join(PROJECT_ROOT, "mod_2_22_no")
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

    # 1. 加载归一化基准
    meta = np.load(STRATEGY_PATH, allow_pickle=True).item()
    norm_meta = meta['norm_meta']
    y_min = torch.tensor(norm_meta['y_min'], dtype=torch.float32).to(DEVICE)
    y_max = torch.tensor(norm_meta['y_max'], dtype=torch.float32).to(DEVICE)

    # 2. 加载测试注册表
    with open(os.path.join(MOD_ROOT, "test_registry.json"), 'r') as f:
        test_registry = json.load(f)

    all_preds_phys = []
    all_trues_phys = []

    # 3. 循环 10 折提取数据
    print("🚀 正在集结 10 折测试集数据...")
    for t_idx in range(1, 11):
        fold_key = f"train_{t_idx}"
        model_path = os.path.join(MOD_ROOT, fold_key, "val_model.pth")

        if not os.path.exists(model_path):
            continue

        model = TwinFusionNet(n_channels=22, n_features=2, n_outputs=44).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        model.eval()

        test_ds = PhysioDataset(test_registry[fold_key], norm_meta, is_train=False)
        loader = DataLoader(test_ds, batch_size=128, shuffle=False)

        with torch.no_grad():
            for w, v, y, _ in loader:
                out_norm = model(w.to(DEVICE), v.to(DEVICE))
                true_norm = y.to(DEVICE)

                # 物理还原
                p_phys = out_norm * (y_max - y_min + 1e-8) + y_min
                t_phys = true_norm * (y_max - y_min + 1e-8) + y_min

                all_preds_phys.append(p_phys.cpu().numpy())
                all_trues_phys.append(t_phys.cpu().numpy())
        print(f"   ✅ Fold {t_idx} 数据提取完成")

    # 垂直堆叠所有折的数据
    preds = np.vstack(all_preds_phys)
    trues = np.vstack(all_trues_phys)

    # ================= 4. 汇总绘图逻辑 (双窗口版 - 纯蓝散点) =================
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # A. 核心逻辑：逐列（分参数）归一化，确保不同量纲参数平等对比
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    # 以真实值为基准训练缩放器
    trues_norm = scaler.fit_transform(trues)
    # 用同样的基准映射预测值
    preds_norm = scaler.transform(preds)

    # B. 拉平所有参数的数据点
    all_t = trues_norm.flatten()
    all_p = preds_norm.flatten()



    # ===============命令行打印 R2 指标 =================
    # 计算所有参数的 R2
    r2_pearson_results = []
    for i in range(len(PARAM_NAMES)):
        t_col = trues_norm[:, i]
        p_col = preds_norm[:, i]
        r2 = r2_score(t_col, p_col)
        try: corr, _ = pearsonr(t_col, p_col)
        except: corr = 0.0
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



    # --------------------- 窗口 1: 散点图 ---------------------------------
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
            ax.set_title(f"{PARAM_NAMES[param_idx]}\n$R^2$={r2:.3f}", fontsize=18)
            ax.tick_params(axis='both', labelsize=16)
            ax.grid(True, linestyle=':', alpha=0.5)

        # 删除该页多余的子图
        for k in range(end_idx - start_idx, len(axes)):
            fig.delaxes(axes[k])

        fig.suptitle(f"44参数详细回归分析 (第 {page + 1}/{pages} 页)", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # ============================================================



    # --- 窗口 2: MAPE 条形图 (保持不变，方便排查具体参数) ---
    plt.figure(num="参数误差排名", figsize=(10, 8))
    mapes = np.mean(np.abs((trues - preds) / (trues + 1e-10)), axis=0) * 100
    sorted_idx = np.argsort(mapes)

    y_pos = np.arange(len(PARAM_NAMES))
    plt.barh(y_pos, mapes[sorted_idx], color='skyblue')
    plt.yticks(y_pos, [PARAM_NAMES[i] for i in sorted_idx], fontsize=9)
    plt.gca().invert_yaxis()

    plt.xlabel("平均绝对百分比误差 (MAPE %)")
    plt.title("44个参数预测误差排名", fontsize=13)
    plt.axvline(x=10, color='red', linestyle='--', alpha=0.5)
    plt.grid(axis='x', linestyle=':', alpha=0.7)
    plt.tight_layout()


    # ===================== 🌟 组合参数分析 🌟 ==============================
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




    print("✨ 分析图已生成")
    plt.show()

    # ===================== 🌟 22通道整合保存逻辑 (绝对路径版) 🌟 ==============================
    # 1. 定义保存目录与绝对路径
    # 根据你的要求，保存在 step_three 的 all_data_no 文件夹下
    SAVE_DIR = r"D:\Digital_twin\step_three\all_data_no"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"📁 已创建文件夹: {SAVE_DIR}")

    final_save_path = os.path.join(SAVE_DIR, "data_22_no.csv")

    # 2. 构造单体参数统计数据 (44个)
    df_single = pd.DataFrame(r2_pearson_results)
    # 使用你代码中计算好的物理还原量 trues 和 preds 计算 MAPE
    single_mapes = np.mean(np.abs((trues - preds) / (trues + 1e-10)), axis=0) * 100
    df_single['mape'] = single_mapes
    df_single.insert(0, '数据类别', '单体参数')

    # 3. 构造组合参数统计数据 (4个)
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

    # 4. 构造总体平均 R2 行 (补齐列以消除 FutureWarning)
    df_avg = pd.DataFrame([{
        '数据类别': '总体统计',
        'name': '所有 44 个参数总平均 R2',
        'r2': avg_r2,
        'corr': 0.0,
        'mape': 0.0
    }])

    # 5. 统一列顺序并纵向合并
    cols = ['数据类别', 'name', 'r2', 'corr', 'mape']
    df_all_stats = pd.concat([df_single[cols], df_comb[cols], df_avg[cols]], ignore_index=True)

    # 6. 保存为 CSV
    df_all_stats.to_csv(final_save_path, index=False, encoding='utf-8-sig')

    print(f"💾 22通道统计结果已成功保存至: {final_save_path}")



if __name__ == "__main__":
    run_cross_val_scatter()