import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter
import numpy as np
from matplotlib.ticker import FixedLocator

# ================= 1. 环境与颜色配置 =================
plt.switch_backend('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 路径锁定 (请确保目录和文件名正确)
DATA_DIR = r"D:\Digital_twin\step_three\all_data"
FILE_11 = os.path.join(DATA_DIR, "RE_01_11.csv")
FILE_22 = os.path.join(DATA_DIR, "RE_02_22.csv")

# 颜色锁定：11通道(蓝色), 22通道(鲜红色)
C1, C2 = "#1a73e8", "#FFA000"
GRAY_COLOR = "#95a5a6"

# 25个物理腔室严格映射序列
ALIGNMENT_MAP = [
    ('R_haa', 'C_haa'), ('R_lna', 'C_lna'), ('R_lca', 'C_lca'), ('R_aop', 'C_aop'),
    ('R_rula', 'C_rula'), ('R_rica', 'C_rica'), ('R_lica', 'C_lica'), ('R_lula', 'C_lula'),
    ('None', 'C_sap'), ('R_rsv', 'C_rsv'), ('R_rijv', 'C_rijv'), ('R_lijv', 'C_lijv'),
    ('R_lsv', 'C_lsv'), ('R_sv', 'C_sv'), ('None', 'C_vc'), ('R_rpap', 'C_rpap'),
    ('R_lpap', 'C_lpap'), ('R_rpad', 'C_rpad'), ('R_lpad', 'C_lpad'), ('R_rpv', 'C_rpv'),
    ('R_lpv', 'C_lpv'), ('R_m', 'None'), ('R_a', 'None'), ('R_t', 'None'), ('R_p', 'None')
]


# ================= 2. 数据处理函数 =================
def get_clean_data(path, p_type='R'):
    if not os.path.exists(path): return pd.DataFrame(), []
    df = pd.read_csv(path)
    raw_order = [it[0] if p_type == 'R' else it[1] for it in ALIGNMENT_MAP]
    unique_order = []
    none_cnt = 0
    for name in raw_order:
        if name == 'None':
            none_cnt += 1
            unique_order.append(f"None_{none_cnt}")
        else:
            unique_order.append(name)
    return df, unique_order


# ================= 3. 核心绘图逻辑 =================
def run_visualization():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 12), facecolor='white')

    # 手动标注“图例”
    fig.text(0.1, 0.96, "■ Model_01 (11-Ch)", color=C1, fontsize=17, fontweight='bold')
    fig.text(0.25, 0.96, "■ Model_02 (22-Ch)", color=C2, fontsize=17, fontweight='bold')

    def draw_layer(ax, file11, file22, p_type, title):
        df11, order = get_clean_data(file11, p_type)
        df22, _ = get_clean_data(file22, p_type)

        for i, param in enumerate(order):
            # 强制 11通道和 22通道并排，但各自内部的 Violin/Box 绝对重合
            pos1, pos2 = i - 0.2, i + 0.2

            # --- 绘制 11通道 (Blue) ---
            if param in df11.columns:
                data1 = df11[param].dropna()
                # 小提琴
                parts1 = ax.violinplot(data1, [pos1], showextrema=False, widths=0.38)
                for pc in parts1['bodies']:
                    pc.set_facecolor(C1)
                    pc.set_alpha(0.3)
                # 箱线图 (强制在 pos1，实现重合)
                ax.boxplot(data1, positions=[pos1], widths=0.1, showfliers=False,
                           patch_artist=True,
                           boxprops=dict(facecolor=C1, alpha=0.8, edgecolor='black'),
                           medianprops=dict(color='white', linewidth=1.5))
                # 离群值标注 (y轴限制在5，所以统计 >5% 的比例)
                ratio = (data1 > 5).sum() / len(data1) * 100
                if ratio > 0.1:
                    ax.text(pos1, 4.6, f"{ratio:.1f}", ha='center', va='bottom',
                            fontsize=12, color=C1, rotation=90, fontweight='bold',
                            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

            # --- 绘制 22通道 (Red) ---
            if param in df22.columns:
                data2 = df22[param].dropna()
                # 小提琴
                parts2 = ax.violinplot(data2, [pos2], showextrema=False, widths=0.38)
                for pc in parts2['bodies']:
                    pc.set_facecolor(C2)
                    pc.set_alpha(0.3)
                # 箱线图 (强制在 pos2，实现重合)
                ax.boxplot(data2, positions=[pos2], widths=0.1, showfliers=False,
                           patch_artist=True,
                           boxprops=dict(facecolor=C2, alpha=0.8, edgecolor='black'),
                           medianprops=dict(color='white', linewidth=1.5))
                # 离群值标注
                ratio = (data2 > 5).sum() / len(data2) * 100
                if ratio > 0.1:
                    ax.text(pos2, 4.6, f"{ratio:.1f}", ha='center', va='bottom',
                            fontsize=12, color=C2, rotation=90, fontweight='bold',
                            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

        # 样式与坐标轴
        ax.set_title(title, fontsize=26, fontweight='bold', pad=10)
        ax.set_ylim(0, 5)  # 🌟 强制 Y 轴限制在 5
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_ylabel("相对误差 (APE %)", fontsize=22)
        ax.axhline(y=0, color='black', linewidth=2)

        ax.tick_params(axis='y', labelsize=18)
        ax.set_xticks(range(len(order)))
        labels = [("" if str(l).startswith('None') else l) for l in order]
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=16, color='black', fontweight='bold')

        # 去掉多余网格，仅保留 y 轴刻度
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    draw_layer(ax1, FILE_11, FILE_22, 'R', "血管阻力 (R) 误差分布统计")
    draw_layer(ax2, FILE_11, FILE_22, 'C', "血管顺应性 (C) 误差分布统计")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    run_visualization()