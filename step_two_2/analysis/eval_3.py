import os
import sys
from pickle import TRUE

import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# 83改分析循环几次的结果
# 93改通道


# ================= 1. 路径与环境强制配置 =================
PROJECT_ROOT = r"D:\Digital_twin\step_two_2"
TRAIN_DIR = os.path.join(PROJECT_ROOT, "train")
MOD_DIR = os.path.join(PROJECT_ROOT, "mod_2_22_no")  # 运行前请确认文件夹名
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "processed_all")

if TRAIN_DIR not in sys.path:
    sys.path.insert(0, TRAIN_DIR)

from model import TwinFusionNet
from dataset import PhysioDataset, DataLoader

# 强制使用交互式后端
matplotlib.use('TkAgg')

# ================= 2. 核心：生理部位对齐蓝图 (Blueprint) =================
# 映射逻辑：(部位名称, 电阻R在44维中的索引, 电容C在44维中的索引)
# R 索引: 0-20 (共21个) | C 索引: 21-43 (共23个)
blueprint = [
    ('二尖瓣(m)', 0, None),
    ('主动脉瓣(a)', 1, None),
    ('三尖瓣(t)', 15, None),
    ('肺动脉瓣(p)', 16, None),

    # ---- 头颈及上肢支路 (R, C 均有) ----
    ('头臂干(haa)', 2, 23),
    ('左颈总(lna)', 3, 24),
    ('左锁骨下(lca)', 4, 25),
    ('近端主动脉(aop)', 5, 26),
    ('右上肢(rula)', 6, 27),
    ('右颈内(rica)', 7, 28),
    ('左颈内(lica)', 8, 29),
    ('左上肢(lula)', 9, 30),

    # ---- 核心体循环下半身 ----
    ('体动脉(sap)', None, 31), # 动态 Rsap 已剔除，只有 Csap

    # ---- 体循环静脉回流 ----
    ('右锁骨下静脉(rsv)', 10, 32),
    ('右颈内静脉(rijv)', 11, 33),
    ('左颈内静脉(lijv)', 12, 34),
    ('左锁骨下静脉(lsv)', 13, 35),
    ('体静脉(sv)', 14, 36),
    ('腔静脉(vc)', None, 37),  # 动态 Rvc 已剔除，只有 Cvc

    # ---- 肺循环 (R, C 均有) ----
    ('右肺动脉近端(rpap)', 17, 38),
    ('左肺动脉近端(lpap)', 18, 39),
    ('右肺动脉远端(rpad)', 19, 40),
    ('左肺动脉远端(lpad)', 20, 41),
    ('右肺静脉(rpv)', 21, 42), # 修复：Rrpv 是有的！
    ('左肺静脉(lpv)', 22, 43)  # 修复：Rlpv 是有的！
]


def eval_loop():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载战略地图和测试注册表
    STRATEGY_PATH = os.path.join(DATA_ROOT, "cv_strategy.npy")
    REGISTRY_PATH = os.path.join(MOD_DIR, "test_registry.json")

    meta = np.load(STRATEGY_PATH, allow_pickle=True).item()
    norm_meta = meta['norm_meta']
    with open(REGISTRY_PATH, 'r') as f:
        test_registry = json.load(f)

    all_preds_phys, all_trues_phys = [], []
    # 🌟 修改点 1：提示信息修改为前 2 轮
    print(f"🚀 正在集结前 2 轮模型进行【全系统对齐阅卷】...")

    # 🌟 修改点 2：将 range(1, 11) 改为 range(1, 3)
    for t_idx in range(1, 11):
        train_key = f"train_{t_idx}"
        model_path = os.path.join(MOD_DIR, train_key, "val_model.pth")
        if not os.path.exists(model_path):
            print(f"⚠️ {train_key} 模型不存在，跳过...")
            continue

        test_ds = PhysioDataset(test_registry[train_key], norm_meta, is_train=False)
        loader = DataLoader(test_ds, batch_size=128, shuffle=False)

        # 初始化 11 通道模型
        model = TwinFusionNet(n_channels=22, n_features=2, n_outputs=44).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        model.eval()

        with torch.no_grad():
            for w, v, y, _ in loader:
                # 模型输出的是 [0, 1] 归一化后的标签
                out_norm = model(w.to(DEVICE), v.to(DEVICE)).cpu().numpy()
                true_norm = y.numpy()

                # 2. 👇 【这里就是 评估反归一化】解除注释，提取极值！
                y_min = norm_meta['y_min']
                y_max = norm_meta['y_max']

                # 3. 将 0~1 的玩具数值，放大回真实的医学物理数值！
                out_phys = out_norm * (y_max - y_min + 1e-8) + y_min
                true_phys = true_norm * (y_max - y_min + 1e-8) + y_min

                all_preds_phys.append(out_phys)
                all_trues_phys.append(true_phys)
        print(f"   ✅ {train_key} 推理并反归一化完成")

    # 汇总计算 APE
    preds = np.concatenate(all_preds_phys, axis=0)
    trues = np.concatenate(all_trues_phys, axis=0)
    ape = np.abs((trues - preds) / (trues + 1e-10)) * 100

    # ================= 终端控制台输出核心评价指标 =================
    valid_r_idx = [r_idx for _, r_idx, _ in blueprint if r_idx is not None]
    valid_c_idx = [c_idx for _, _, c_idx in blueprint if c_idx is not None]

    mean_ape_per_param = np.nanmean(ape, axis=0)
    overall_mape = np.nanmean(ape)

    total_params = ape.shape[1]
    count_under_10 = np.sum(mean_ape_per_param < 10)

    r_mape = np.nanmean(ape[:, valid_r_idx])
    c_mape = np.nanmean(ape[:, valid_c_idx])

    print("\n" + "=" * 40)
    # 🌟 修改点 3：指标打印加入 [前2轮]
    print(f"评估结果统计 —— 11通道输入 (Mod_11) [前2轮]")
    print(f"平均相对误差 (MAPE): {overall_mape:.2f}%")
    print(f"误差 < 10%: {(count_under_10 / total_params) * 100:.2f}% ({count_under_10}/{total_params})")
    print(f"电阻组 (R) 平均误差: {r_mape:.2f}%")
    print(f"电容组 (C) 平均误差: {c_mape:.2f}%")
    print("=" * 40 + "\n")

    # ================= 3. 数据重组 =================
    r_plot_data, r_labels = [], []
    c_plot_data, c_labels = [], []

    for name, r_idx, c_idx in blueprint:
        if r_idx is not None:
            r_plot_data.append(ape[:, r_idx])
            r_labels.append(f"R_{name}")
        else:
            r_plot_data.append(np.array([np.nan]))
            r_labels.append("")

        if c_idx is not None:
            c_plot_data.append(ape[:, c_idx])
            c_labels.append(f"C_{name}")
        else:
            c_plot_data.append(np.array([np.nan]))
            c_labels.append("")

    # ================= 4. 绘图：小提琴 + 箱线图 =================
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))

    def draw_complex_plot(ax, data, labels, title, color):
        all_valid_data = []
        plot_positions = []
        plot_data = []

        for i, d in enumerate(data):
            valid_d = d[~np.isnan(d)]

            if len(valid_d) > 0:
                all_valid_data.extend(valid_d)
                plot_positions.append(i)
                plot_data.append(valid_d)

                ptp = np.max(valid_d) - np.min(valid_d)
                if ptp > 1e-5:
                    vparts = ax.violinplot(valid_d, positions=[i], showextrema=False, widths=0.7, vert=True, points=100)
                    for pc in vparts['bodies']:
                        pc.set_facecolor(color)
                        pc.set_alpha(0.3)
                else:
                    ax.scatter([i], [np.median(valid_d)], color='red', s=40, zorder=4, marker='*')

        if plot_data:
            ax.boxplot(plot_data, positions=plot_positions, widths=0.12, showfliers=False,
                       patch_artist=True, boxprops=dict(facecolor=color, alpha=0.7),
                       medianprops=dict(color='yellow', linewidth=2))

        ax.set_title(title, fontsize=28, fontweight='bold')
        ax.set_ylabel('相对误差 (APE %)', fontsize=22, fontweight='bold')
        ax.axhline(y=10, color='red', linestyle='--', alpha=0.6, label='10% 临床接受阈值')
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=16)
        ax.tick_params(axis='y', labelsize=16)

        if len(all_valid_data) > 0:
            p98 = np.percentile(all_valid_data, 98)
            pmax = np.max(all_valid_data)
            upper_limit = min(pmax * 1.1, p98 * 1.3)
            ax.set_ylim(-2, max(15, upper_limit))

        ax.grid(axis='y', linestyle=':', alpha=0.5)
        ax.legend(loc='upper right', fontsize=16)

    # 🌟 修改点 4：更新图表标题
    draw_complex_plot(ax1, r_plot_data, r_labels, '电阻 (R) 参数评估误差 - 22通道对齐汇总', 'steelblue')
    draw_complex_plot(ax2, c_plot_data, c_labels, '电容 (C) 参数评估误差 - 22通道对齐汇总', 'darkorange')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    eval_loop()