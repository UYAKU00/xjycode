import os
import glob
import json
import numpy as np
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib import ticker

# 强行锁定交互式后端
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ================= 1. 路径与环境配置 =================
PROJECT_ROOT = r"D:\Digital_twin\step_one"
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset_10000")

# ================= 2. 统一大蓝图 (完美对齐顺序) =================
unified_blueprint = [
    ('m', '二尖瓣', 0, 0.015, None, None),
    ('a', '主动脉瓣', 1, 0.02, None, None),
    ('t', '三尖瓣', 17, 0.02, None, None),
    ('p', '肺动脉瓣', 18, 0.01, None, None),
    ('haa', '头臂干', 2, 13, 0, 1),
    ('lna', '左颈总', 3, 16, 1, 1),
    ('lca', '左锁骨下', 4, 16, 2, 1),
    ('aop', '近端主动脉', 5, 1.2, 3, 0.8),
    ('rula', '右上肢', 6, 0.4, 4, 3),
    ('rica', '右颈内', 7, 0.4, 5, 2),
    ('lica', '左颈内', 8, 0.4, 6, 4),
    ('lula', '左上肢', 9, 0.4, 7, 2),
    ('sap', '体动脉', None, None, 8, 5),
    ('rsv', '右锁骨下静脉', 11, 0.17, 9, 10),
    ('rijv', '右颈内静脉', 12, 0.2, 10, 10),
    ('lijv', '左颈内静脉', 13, 0.2, 11, 10),
    ('lsv', '左锁骨下静脉', 14, 0.2, 12, 10),
    ('sv', '体静脉', 15, 0.2, 13, 20),
    ('vc', '腔静脉', None, None, 14, 30),
    ('rpap', '右肺近端', 19, 0.02, 15, 10),
    ('lpap', '左肺近端', 20, 0.02, 16, 10),
    ('rpad', '右肺远端', 21, 0.03, 17, 23),
    ('lpad', '左肺远端', 22, 0.03, 18, 23),
    ('rpv', '右肺静脉', 23, 0.045, 19, 25),
    ('lpv', '左肺静脉', 24, 0.045, 20, 25)
]

SHARED_LABELS = [f"{eng}\n({zh})" for eng, zh, _, _, _, _ in unified_blueprint]


# ================= 3. 数据提取 =================
def load_all_json_data():
    print(f"📂 正在扫描目录: {DATASET_DIR}")
    json_pattern = os.path.join(DATASET_DIR, "**", "inputs.json")
    all_json_files = glob.glob(json_pattern, recursive=True)
    if not all_json_files: return np.array([]), np.array([])
    R_list, C_list = [], []
    for file in all_json_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'R' in data: R_list.append(data['R'])
                if 'C' in data: C_list.append(data['C'])
        except:
            continue
    return np.array(R_list), np.array(C_list)


def extract_unified_data(data_matrix, is_r=True):
    extracted_data, true_values = [], []
    for item in unified_blueprint:
        idx = item[2] if is_r else item[4]
        true_val = item[3] if is_r else item[5]
        true_values.append(true_val)
        if idx is not None and data_matrix.size > 0 and idx < data_matrix.shape[1]:
            extracted_data.append(data_matrix[:, idx])
        else:
            extracted_data.append(np.array([np.nan]))
    return extracted_data, true_values


# ================= 4. 辅助函数 =================
def get_dynamic_limits(data_arrays, true_values, indices):
    if not indices: return 0, 1
    max_val, min_val, has_data = -np.inf, np.inf, False
    for i in indices:
        tv = true_values[i]
        if tv is None: continue
        d = data_arrays[i]
        valid_d = d[~np.isnan(d)]
        if len(valid_d) > 0:
            varied_d = valid_d[np.abs(valid_d - tv) / tv > 0.001]
            if len(varied_d) > 0:
                max_val, min_val = max(max_val, np.max(varied_d)), min(min_val, np.min(varied_d))
                has_data = True
            else:
                max_val, min_val, has_data = max(max_val, tv), min(min_val, tv), True
        else:
            max_val, min_val, has_data = max(max_val, tv), min(min_val, tv), True

    if not has_data: return 0, 1
    range_val = max_val - min_val
    if range_val == 0: range_val = max_val * 0.2 if max_val != 0 else 1.0
    return min_val - range_val * 0.05, max_val + range_val * 0.15


def draw_break_lines(ax_top, ax_bottom):
    d = 0.012
    kwargs = dict(color='black', clip_on=False, linewidth=1.5)
    ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), transform=ax_top.transAxes, **kwargs)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), transform=ax_bottom.transAxes, **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_bottom.transAxes, **kwargs)


# ================= 5. 分层截断与连桥渲染引擎 =================
def plot_dataset_distribution():
    R_matrix, C_matrix = load_all_json_data()
    r_data, R_TRUE = extract_unified_data(R_matrix, is_r=True)
    c_data, C_TRUE = extract_unified_data(C_matrix, is_r=False)

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(24, 18))

    # 🌟 扩充间距：将 top 和 bottom 拉开，避免上下子图打架
    # R 组稍微往上收 (bottom=0.57)
    gs_r = gridspec.GridSpec(4, 1, figure=fig, top=0.95, bottom=0.57, height_ratios=[1.5, 1, 1, 2], hspace=0.1)
    # C 组稍微往下压 (top=0.43) -> 缓冲带从原本的 0.07 增加到了 0.14！
    gs_c = gridspec.GridSpec(3, 1, figure=fig, top=0.43, bottom=0.06, height_ratios=[1.5, 1, 2], hspace=0.1)

    def draw_tiered_plot(gridspec_obj, data_arrays, true_values, title, color, violin_color, layer_mapping, ylimits,
                         ylabel_pos):
        axes = [fig.add_subplot(gs) for gs in gridspec_obj]

        for ax_idx, ax in enumerate(axes):
            ymin, ymax = ylimits[ax_idx]

            for i, (d, true_val) in enumerate(zip(data_arrays, true_values)):
                if true_val is None: continue
                valid_d = d[~np.isnan(d)]

                if true_val > 0 and true_val >= ymin:
                    ax.bar(i, true_val, width=0.6, facecolor=color, alpha=0.35, edgecolor='black', linewidth=0.8,
                           zorder=2)

                    if true_val > ymax and ax_idx > 0:
                        ax.fill_between([i - 0.3, i + 0.3], 1.0, 1.35, transform=ax.get_xaxis_transform(),
                                        clip_on=False, facecolor=color, alpha=0.35, zorder=1)
                        ax.plot([i - 0.3, i - 0.3], [1.0, 1.35], transform=ax.get_xaxis_transform(), clip_on=False,
                                color='black', lw=0.8, zorder=2)
                        ax.plot([i + 0.3, i + 0.3], [1.0, 1.35], transform=ax.get_xaxis_transform(), clip_on=False,
                                color='black', lw=0.8, zorder=2)

                    if len(valid_d) > 0:
                        variation = np.abs(valid_d - true_val) / true_val
                        varied_d = valid_d[variation > 0.001]
                        if len(varied_d) > 0:
                            top_layer_idx = 0
                            for idx, mapping in enumerate(layer_mapping):
                                if i in mapping: top_layer_idx = idx; break

                            if ax_idx == top_layer_idx:
                                vparts = ax.violinplot(varied_d, positions=[i], showextrema=False, widths=0.4,
                                                       vert=True)
                                for pc in vparts['bodies']:
                                    pc.set_facecolor(violin_color)
                                    pc.set_edgecolor('black')
                                    pc.set_linewidth(1.0)
                                    pc.set_alpha(0.8)
                                    pc.set_zorder(3)

                                ax.boxplot(varied_d, positions=[i], widths=0.1, showfliers=False, patch_artist=True,
                                           boxprops=dict(facecolor='white', edgecolor='black', linewidth=0.8, alpha=0.6,
                                                         zorder=4),
                                           medianprops=dict(color='#D32F2F', linewidth=2, zorder=5),
                                           whiskerprops=dict(color='black', linewidth=0.8, alpha=0.8),
                                           capprops=dict(color='black', linewidth=0.8, alpha=0.8))

            ax.set_xlim(-0.6, len(SHARED_LABELS) - 0.4)
            ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
            for x in range(len(SHARED_LABELS)):
                ax.axvline(x=x, color='gray', linestyle=':', alpha=0.15, zorder=0)

        # 🌟 字号全面缩小：由 10 降为 8.5
        for i, true_val in enumerate(true_values):
            if true_val is not None:
                layer_idx = 0
                for idx, mapping in enumerate(layer_mapping):
                    if i in mapping:
                        layer_idx = idx;
                        break

                target_ax = axes[layer_idx]
                d = data_arrays[i]
                valid_d = d[~np.isnan(d)]
                text_y = true_val
                if len(valid_d) > 0:
                    varied_d = valid_d[np.abs(valid_d - true_val) / true_val > 0.001]
                    if len(varied_d) > 0: text_y = max(np.max(varied_d), true_val)

                ymin, ymax = ylimits[layer_idx]
                offset = (ymax - ymin) * 0.04

                # fontsize=8.5 更加精致
                target_ax.text(i, text_y + offset, f"{true_val:.3f}", ha='center', va='bottom',
                               fontsize=8.5, fontweight='normal', color='black', zorder=10)

        for i, ax in enumerate(axes):
            ax.set_ylim(*ylimits[i])
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
            if i < len(axes) - 1:
                ax.spines['bottom'].set_color('none')
                ax.xaxis.tick_top()
                ax.tick_params(labeltop=False, bottom=False)
                draw_break_lines(axes[i], axes[i + 1])
            if i > 0:
                ax.spines['top'].set_color('none')
                ax.xaxis.tick_bottom()

        # 🌟 标题字号：由 18 降为 15
        axes[0].set_title(title, fontsize=15, fontweight='bold', pad=12)
        axes[-1].set_xticks(np.arange(len(SHARED_LABELS)))
        # 🌟 X轴刻度字号：由 10 降为 8.5
        axes[-1].set_xticklabels(SHARED_LABELS, rotation=30, ha='right', fontsize=8.5, fontweight='bold')

        # 🌟 Y轴整体标签：由 14 降为 12
        fig.text(0.08, ylabel_pos, '真实物理值 (从 0 起步)', va='center', rotation='vertical', fontsize=12,
                 fontweight='bold')

    # ================= 6. 集群分类 =================
    idx_r_l0 = [i for i, tv in enumerate(R_TRUE) if tv is not None and tv >= 10]
    idx_r_l1 = [i for i, tv in enumerate(R_TRUE) if tv is not None and 1 <= tv < 10]
    idx_r_l2 = [i for i, tv in enumerate(R_TRUE) if tv is not None and 0.3 <= tv < 1]
    idx_r_l3 = [i for i, tv in enumerate(R_TRUE) if tv is not None and tv < 0.3]

    y_r0 = get_dynamic_limits(r_data, R_TRUE, idx_r_l0)
    y_r1 = get_dynamic_limits(r_data, R_TRUE, idx_r_l1)
    y_r2 = get_dynamic_limits(r_data, R_TRUE, idx_r_l2)
    _, ymax_r3 = get_dynamic_limits(r_data, R_TRUE, idx_r_l3)

    idx_c_l0 = [i for i, tv in enumerate(C_TRUE) if tv is not None and tv >= 18]
    idx_c_l1 = [i for i, tv in enumerate(C_TRUE) if tv is not None and 6 <= tv < 18]
    idx_c_l2 = [i for i, tv in enumerate(C_TRUE) if tv is not None and tv < 6]

    y_c0 = get_dynamic_limits(c_data, C_TRUE, idx_c_l0)
    y_c1 = get_dynamic_limits(c_data, C_TRUE, idx_c_l1)
    _, ymax_c2 = get_dynamic_limits(c_data, C_TRUE, idx_c_l2)

    # ================= 7. 渲染出图 =================
    draw_tiered_plot(gs_r, r_data, R_TRUE,
                     title='10,000组仿真数据 电阻(R) 分布 (四层断轴・无缝连通)',
                     color='#5DADE2', violin_color='#4682B4',
                     layer_mapping=[idx_r_l0, idx_r_l1, idx_r_l2, idx_r_l3],
                     ylimits=[y_r0, y_r1, y_r2, (0, ymax_r3)],
                     ylabel_pos=0.75)

    draw_tiered_plot(gs_c, c_data, C_TRUE,
                     title='10,000组仿真数据 电容(C) 分布 (三层断轴・无缝连通)',
                     color='#F5B041', violin_color='#D35400',
                     layer_mapping=[idx_c_l0, idx_c_l1, idx_c_l2],
                     ylimits=[y_c0, y_c1, (0, ymax_c2)],
                     ylabel_pos=0.25)

    save_path = os.path.join(PROJECT_ROOT, "Dataset_10000_BrokenAxis_ShrinkFont.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 字号瘦身排版修复大图已生成并保存至: {save_path}")

    plt.show(block=True)


if __name__ == "__main__":
    plot_dataset_distribution()