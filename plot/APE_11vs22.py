import matplotlib

# 强制使用 TkAgg 后端以确保弹出独立窗口
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ================= 1. 环境配置 =================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ================= 2. 解剖顺序参数序列 (根据表 3.1) =================
# 阻力参数 R 序列
params_R = [
    "R_aop", "R_haa", "R_lna", "R_lca", "R_rula", "R_lula", "R_rica", "R_lica", "R_sap",
    "R_rijv", "R_lijv", "R_rsv", "R_lsv", "R_sv", "R_vc",
    "R_rpap", "R_lpap", "R_rpad", "R_lpad", "R_rpv", "R_lpv", "R_m", "R_a", "R_t", "R_p"
]

# 顺应性参数 C 序列
params_C = [
    "C_aop", "C_haa", "C_lna", "C_lca", "C_rula", "C_lula", "C_rica", "C_lica", "C_sap",
    "C_rijv", "C_lijv", "C_rsv", "C_lsv", "C_sv", "C_vc",
    "C_rpap", "C_lpap", "C_rpad", "C_lpad", "C_rpv", "C_lpv"
]

# ================= 3. 数据填入 (请根据您的表格准确填入) =================

# --- 11-Ch (Model_01) 数据 ---
# R2
r2_R_11 = [0.9125, 0.9506, 0.9468, 0.9616, 0.8012, 0.7340, 0.7390, -0.0091, 0.7058, 0.1398, -0.0188, 0.1602, 0.2247,
           0.7969, 0.0166, -0.0303, -0.0408, -0.0181, -0.0287, 0.1112, 0.1074, 0.7058, -0.0519, 0.9143, -0.0283]
r2_C_11 = [0.3992, 0.8102, 0.5047, 0.5581, 0.0263, 0.0029, -0.0606, -0.2717, -0.3390, 0.2455, 0.1536, 0.1009, 0.1883,
           -0.0024, 0.0102, 0.0306, 0.0282, 0.1259, 0.1146, 0.1754, 0.1664]
# r
corr_R_11 = [0.9697, 0.9845, 0.9861, 0.9878, 0.9037, 0.8703, 0.8646, 0.0877, 0.8489, 0.3992, 0.0900, 0.4175, 0.4800,
             0.9013, 0.1914, 0.0292, 0.0081, 0.0915, 0.0685, 0.3506, 0.3619, 0.8489, 0.0038, 0.9654, 0.0746]
corr_C_11 = [0.6650, 0.9198, 0.7428, 0.7757, 0.3191, 0.2452, 0.1285, 0.2403, 0.1728, 0.4983, 0.4313, 0.3224, 0.4354,
             0.1404, 0.1713, 0.1904, 0.1937, 0.3629, 0.3599, 0.4250, 0.4100]
# MAPE
mape_R_11 = [0.97] * 25
mape_C_11 = [0.85] * 21

# --- 22-Ch (Model_02) 数据 ---
# R2
r2_R_22 = [0.9603, 0.9760, 0.9742, 0.9748, 0.9723, 0.8929, 0.8088, 0.1091, 0.9579, 0.9311, 0.2618, 0.9428, 0.7101,
           0.9277, 0.0102, -0.0131, -0.0231, -0.0006, -0.0033, 0.3645, 0.3773, 0.9579, -0.0334, 0.9655, -0.0145]
r2_C_22 = [0.8399, 0.7910, 0.9707, 0.6981, 0.0263, 0.0371, 0.9484, 0.8043, -0.1031, 0.2455, 0.4198, 0.6488, 0.2211,
           0.0220, 0.0102, 0.0586, 0.0475, 0.1259, 0.1146, 0.1754, 0.1664]
# r
corr_R_22 = [0.9820, 0.9906, 0.9913, 0.9911, 0.9875, 0.9478, 0.9040, 0.3426, 0.9811, 0.9675, 0.5234, 0.9766, 0.8457,
             0.9670, 0.1713, 0.0521, 0.0197, 0.1030, 0.0856, 0.6113, 0.6185, 0.9811, -0.0047, 0.9880, 0.0575]
corr_C_22 = [0.9199, 0.8943, 0.9909, 0.8486, 0.3191, 0.2564, 0.9794, 0.9167, 0.1936, 0.4983, 0.6805, 0.8217, 0.4724,
             0.1679, 0.1713, 0.2458, 0.2306, 0.3629, 0.3599, 0.4250, 0.4100]
# MAPE
mape_R_22 = [0.90] * 25
mape_C_22 = [0.78] * 21


# ================= 4. 绘图函数 (上下子图结构) =================

def create_double_plot(window_title, metric_name, data_r_11, data_r_22, data_c_11, data_c_22):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    fig.canvas.manager.set_window_title(window_title)

    # 上子图: 阻力参数 R
    ax1.plot(params_R, data_r_11, marker='o', label='11-Ch', color='#1a73e8', linewidth=2)
    ax1.plot(params_R, data_r_22, marker='s', label='22-Ch', color='#FFA000', linewidth=2)
    ax1.set_ylabel(f"{metric_name} Value", fontsize=22, fontweight='bold')
    ax1.set_title(f"阻力参数 (R) 辨识精度对比 - {metric_name}", fontsize=27, fontweight='bold')
    ax1.legend(prop={'size': 15})
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax1.tick_params(axis='x', rotation=90, labelsize=18)
    ax1.tick_params(axis='y', labelsize=18)

    # 下子图: 顺应性参数 C
    ax2.plot(params_C, data_c_11, marker='o', label='11-Ch', color='#1a73e8', linewidth=2)
    ax2.plot(params_C, data_c_22, marker='s', label='22-Ch', color='#FFA000', linewidth=2)
    ax2.set_ylabel(f"{metric_name} Value", fontsize=22, fontweight='bold')
    ax2.set_title(f"顺应性参数 (C) 辨识精度对比 - {metric_name}", fontsize=27, fontweight='bold')
    ax2.legend(prop={'size': 15})
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax2.tick_params(axis='x', rotation=90, labelsize=18)
    ax2.tick_params(axis='y', labelsize=18)

    plt.tight_layout()


# ================= 5. 执行弹出三个窗口 =================

# 窗口 1: R2
create_double_plot("决定系数 R2 对比图", "R^2", r2_R_11, r2_R_22, r2_C_11, r2_C_22)

# 窗口 2: 相关系数 r
create_double_plot("相关系数 r 对比图", "r", corr_R_11, corr_R_22, corr_C_11, corr_C_22)

plt.show()