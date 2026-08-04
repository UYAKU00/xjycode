import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import os
import matplotlib

# 强制锁定 TkAgg 后端
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ================= 1. 路径配置 =================
OUR_MODEL_RE_PATH = r'D:\Digital_twin\step_three\all_data_loss\RE_02_22.csv'
BASE_MODEL_RE_PATH = r'D:\Digital_twin\step_three\all_data_loss\RE_01_11.csv'

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================= 2. 解剖位置蓝图 =================
blueprint = [
    ('m', '二尖瓣'), ('a', '主动脉瓣'), ('t', '三尖瓣'), ('p', '肺动脉瓣'),
    ('haa', '头臂干'), ('lna', '左颈总'), ('lca', '左锁骨下'), ('aop', '近端主动脉'),
    ('rula', '右上肢'), ('rica', '右颈内'), ('lica', '左颈内'), ('lula', '左上肢'),
    ('sap', '体动脉'), ('rsv', '右锁骨下静脉'), ('rijv', '右颈内静脉'),
    ('lijv', '左颈内静脉'), ('lsv', '左锁骨下静脉'), ('sv', '体静脉'),
    ('vc', '腔静脉'), ('rpap', '右肺动脉近端'), ('lpap', '左肺动脉近端'),
    ('rpad', '右肺动脉远端'), ('lpad', '左肺动脉远端'), ('rpv', '右肺静脉'),
    ('lpv', '左肺静脉')
]


def compute_data_and_sig(df_ours, df_base, prefix):
    m_ours, m_base, sig_p_vals = [], [], []
    for eng, zh in blueprint:
        col = f"{prefix}_{eng}"
        if col in df_ours.columns:
            d1, d2 = df_ours[col].values, df_base[col].values
            m1, m2 = np.mean(np.abs(d1)), np.mean(np.abs(d2))
            if np.array_equal(d1, d2):
                p = 1.0
            else:
                _, p = wilcoxon(d1, d2, alternative='two-sided')
            m_ours.append(m1)
            m_base.append(m2)
            sig_p_vals.append(p)
        else:
            m_ours.append(None)
            m_base.append(None)
            sig_p_vals.append(None)
    return m_ours, m_base, sig_p_vals


def plot_final_independent_window():
    if not os.path.exists(OUR_MODEL_RE_PATH) or not os.path.exists(BASE_MODEL_RE_PATH):
        print("❌ 错误：路径不存在，请检查 CSV 文件位置。")
        return

    df_22 = pd.read_csv(OUR_MODEL_RE_PATH)
    df_11 = pd.read_csv(BASE_MODEL_RE_PATH)

    r_ours, r_base, r_sig_p = compute_data_and_sig(df_22, df_11, "R")
    c_ours, c_base, c_sig_p = compute_data_and_sig(df_22, df_11, "C")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))
    fig.canvas.manager.set_window_title('心肺系统血流动力学参数预测对比 - 显著性增强分析')

    x = np.arange(len(blueprint))
    width = 0.38
    labels = [f"{eng}\n({zh})" for eng, zh in blueprint]

    def draw_sub(ax, ours, base, p_vals, title, c1, c2):
        o_plot = [v if v is not None else 0 for v in ours]
        b_plot = [v if v is not None else 0 for v in base]

        ax.bar(x - width / 2, o_plot, width, label='22_channel (Ours)', color=c1, edgecolor='black', lw=0.6)
        ax.bar(x + width / 2, b_plot, width, label='11_channel (Base)', color=c2, edgecolor='black', lw=0.6)

        # --- 核心改进：条件标注星号 ---
        for i in range(len(p_vals)):
            p = p_vals[i]
            if p is not None and p < 0.05:
                h = max(o_plot[i], b_plot[i])
                # 判断 22通道(ours) 是否比 11通道(base) 表现更好
                if o_plot[i] < b_plot[i]:
                    star_color = 'red'  # 性能显著优化（误差降低）
                else:
                    star_color = 'orange'  # 性能显著变差（误差升高）

                ax.text(x[i], h + (h * 0.01), '*', ha='center', va='bottom',
                        color=star_color, fontsize=20, fontweight='bold')

        ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
        ax.set_ylabel('平均相对误差 (MAPE)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', linestyle=':', alpha=0.6)

        # 增加图例说明星号含义
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color='red', marker='*', linestyle='None', markersize=10,
                   label='显著优化 (p<0.05, Ours < Base)'),
            Line2D([0], [0], color='orange', marker='*', linestyle='None', markersize=10,
                   label='显著变差 (p<0.05, Ours > Base)')
        ]
        current_handles, current_labels = ax.get_legend_handles_labels()
        ax.legend(handles=current_handles + custom_lines, loc='upper right', frameon=True, fontsize=10)

        ax.set_xlim(-0.8, len(blueprint) - 0.2)

    draw_sub(ax1, r_ours, r_base, r_sig_p, "血管阻力参数评估 (R)：11通道 vs 22通道", "#2077B4", "#AEC7E8")
    draw_sub(ax2, c_ours, c_base, c_sig_p, "血管顺应性参数评估 (C)：11通道 vs 22通道", "#D62728", "#FF9896")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)

    print("🚀 独立窗口已弹出。红色星号代表显著变好，黄色/橙色代表显著变差。")
    plt.show(block=True)


if __name__ == "__main__":
    plot_final_independent_window()