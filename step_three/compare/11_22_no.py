import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.ticker import FixedLocator

# ================= 1. 环境与路径配置 =================
matplotlib.use('TkAgg')
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = r"D:\Digital_twin\step_three\all_data_no"
FILE_11 = os.path.join(DATA_DIR, "data_11_no.csv")
FILE_22 = os.path.join(DATA_DIR, "data_22_no.csv")

# 统一标签与配色 (香蕉黄 #FFE135)
L1, L2 = 'Model_01 (11-Ch)', 'Model_02 (22-Ch)'
STRIKING_PALETTE = {L1: "#1a73e8", L2: "#FFE135"}
GRAY_COLOR = "#95a5a6"

# ================= 2. 严格 25 个物理腔室对齐序列 =================
ALIGNMENT_MAP = [
    ('R_haa', 'C_haa'), ('R_lna', 'C_lna'), ('R_lca', 'C_lca'), ('R_aop', 'C_aop'),
    ('R_rula', 'C_rula'), ('R_rica', 'C_rica'), ('R_lica', 'C_lica'), ('R_lula', 'C_lula'),
    ('R_sap', 'C_sap'), ('R_rsv', 'C_rsv'), ('R_rijv', 'C_rijv'), ('R_lijv', 'C_lijv'),
    ('R_lsv', 'C_lsv'), ('R_sv', 'C_sv'), ('R_vc', 'C_vc'),
    ('R_rpap', 'C_rpap'), ('R_lpap', 'C_lpap'), ('R_rpad', 'C_rpad'), ('R_lpad', 'C_lpad'),
    ('R_rpv', 'C_rpv'), ('R_lpv', 'C_lpv'),
    ('R_m', 'None'), ('R_a', 'None'), ('R_t', 'None'), ('R_p', 'None')
]


# ================= 3. 通用样式处理函数 =================
def apply_unified_style(ax, title, ylabel, ylim):
    """统一设置标题、标签、刻度、负数显示等"""
    ax.set_title(title, fontsize=16, fontweight='bold', pad=5, color=GRAY_COLOR)
    ax.set_ylabel(ylabel, fontsize=15, fontweight='bold', color=GRAY_COLOR)
    ax.set_xlabel("")
    ax.axhline(y=0, color='black', linewidth=1.5)
    ax.set_ylim(ylim)

    # 强制固定刻度并清理 None 标签
    ax.xaxis.set_major_locator(FixedLocator(np.arange(len(ALIGNMENT_MAP))))
    current_labels = [label.get_text() for label in ax.get_xticklabels()]
    clean_labels = [l.split('_')[0] if 'None' in str(l) else l for l in current_labels]
    ax.set_xticklabels(clean_labels, rotation=45, ha='right',
                       fontsize=9, fontweight='bold', color=GRAY_COLOR)


# ================= 4. 数据加载 (适配 R2 和 APE% 和 r) =================
def load_aligned_data(path, model_label, param_type='R'):
    df_raw = pd.read_csv(path)
    df_raw = df_raw[df_raw['数据类别'] == '单体参数'].copy()

    # 确定取左列(R)还是右列(C)
    full_order = [item[0] if param_type == 'R' else item[1] for item in ALIGNMENT_MAP]

    aligned_list = []
    none_count = 0
    for p_name in full_order:
        if p_name == 'None':
            none_count += 1
            aligned_list.append(
                {'name': f"None_{none_count}", 'r2': np.nan, 'mape': np.nan, 'corr': np.nan, 'Model': model_label})
        else:
            row = df_raw[df_raw['name'] == p_name]
            r2 = row['r2'].values[0] if not row.empty else np.nan
            mape = row['mape'].values[0] if not row.empty and 'mape' in row.columns else np.nan
            # 🌟 关键修复：这里的列名必须是 'corr'
            pearson = row['corr'].values[0] if not row.empty and 'corr' in row.columns else np.nan

            aligned_list.append({'name': p_name, 'r2': r2, 'mape': mape, 'corr': pearson, 'Model': model_label})
    return pd.DataFrame(aligned_list)

# 加载数据
df_r = pd.concat([load_aligned_data(FILE_11, L1, 'R'), load_aligned_data(FILE_22, L2, 'R')], ignore_index=True)
df_c = pd.concat([load_aligned_data(FILE_11, L1, 'C'), load_aligned_data(FILE_22, L2, 'C')], ignore_index=True)


# =================🌟 5. 绘图 Figure 1: R2 Score 🌟=================
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 12))
fig1.canvas.manager.set_window_title('R2 精度对比图')

sns.barplot(ax=ax1, data=df_r, x='name', y='r2', hue='Model', palette=STRIKING_PALETTE, edgecolor='black',
            linewidth=0.8, width=0.6)
apply_unified_style(ax1, "血管阻力 (R) R^2 对比", "R^2 Score", (-0.5, 1.0))
ax1.axhline(y=0.8, color='red', linestyle='--', linewidth=1.2, alpha=0.5)

sns.barplot(ax=ax2, data=df_c, x='name', y='r2', hue='Model', palette=STRIKING_PALETTE, edgecolor='black',
            linewidth=0.8, width=0.6)
apply_unified_style(ax2, "血管顺应性 (C) R^2 对比", "R^2 Score", (-0.5, 1.0))
ax2.axhline(y=0.8, color='red', linestyle='--', linewidth=1.2, alpha=0.5)

ax1.legend(loc='upper right', frameon=True).get_texts()[0].set_color(GRAY_COLOR)
ax2.get_legend().remove()
fig1.tight_layout()



# =================🌟 6. 绘图 Figure 2: Pearson 相关系数 (r) 对比 🌟=================
fig2, (ax5, ax6) = plt.subplots(2, 1, figsize=(22, 12))
fig2.canvas.manager.set_window_title('Pearson 相关系数对比图')
sns.barplot(ax=ax5, data=df_r, x='name', y='corr', hue='Model', palette=STRIKING_PALETTE, edgecolor='black', linewidth=0.8, width=0.6)
apply_unified_style(ax5, "血管阻力 (R) Pearson 相关系数对比", "Pearson r", (-0.1, 1.05))
ax5.axhline(y=0.8, color='red', linestyle='--', linewidth=1.2, alpha=0.5)

sns.barplot(ax=ax6, data=df_c, x='name', y='corr', hue='Model', palette=STRIKING_PALETTE, edgecolor='black', linewidth=0.8, width=0.6)
apply_unified_style(ax6, "血管顺应性 (C) Pearson 相关系数对比", "Pearson r", (-0.1, 1.05))
ax6.axhline(y=0.8, color='red', linestyle='--', linewidth=1.2, alpha=0.5)
ax5.legend(loc='upper right', frameon=True).get_texts()[0].set_color(GRAY_COLOR)
ax6.get_legend().remove()
fig2.tight_layout()



plt.show()