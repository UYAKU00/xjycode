import matplotlib
matplotlib.use('TkAgg') # 确保独立弹窗
import matplotlib.pyplot as plt

# ================= 1. 配置与数据 =================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 四组组合参数标签
groups = ["远端系统阻力", "静脉系统总顺应性", "肺循环总阻力", "肺循环总顺应性"]

# --- 11通道 (Model_01) 数据 ---
r2_11 = [-0.0203,0.0994, -0.0575, 0.0664] # 示例数据，请根据实际表格修改
r_11 = [0.0858, 0.3996, 0.1822, 0.2922]

# --- 22通道 (Model_02) 数据 ---
r2_22 = [0.2221, 0.1193, -0.0002, 0.1801]
r_22 = [0.4850, 0.3613, 0.1672, 0.4395]

# ================= 2. 绘图逻辑 =================

# 窗口 1: R2 对比
plt.figure("组合参数 R2 对比", figsize=(10, 6))
plt.plot(groups, r2_11, marker='o', label='11-Ch (Model_01)', color='#1a73e8', linewidth=2.5, markersize=8)
plt.plot(groups, r2_22, marker='s', label='22-Ch (Model_02)', color='#FFA000', linewidth=2.5, markersize=8)
plt.ylabel("$R^2$ Value", fontsize=18)
plt.title("组合生理参数辨识精度对比 - $R^2$", fontsize=26, fontweight='bold')
plt.ylim(-0.1, 0.8) # 组合参数精度通常较高，坐标轴可适当上移
plt.legend(prop={'size': 16})
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.xticks(fontsize=18)
plt.tick_params(axis='y', labelsize=17)
plt.tight_layout()

# 窗口 2: 相关系数 r 对比
plt.figure("组合参数 r 对比", figsize=(10, 6))
plt.plot(groups, r_11, marker='o', label='11-Ch (Model_01)', color='#1a73e8', linewidth=2.5, markersize=8)
plt.plot(groups, r_22, marker='s', label='22-Ch (Model_02)', color='#FFA000', linewidth=2.5, markersize=8)
plt.ylabel("相关系数 r", fontsize=18)
plt.title("组合生理参数辨识精度对比 - r", fontsize=26, fontweight='bold')
plt.ylim(-0.1, 0.8)
plt.legend(prop={'size': 16})
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.xticks(fontsize=18)
plt.tick_params(axis='y', labelsize=17)
plt.tight_layout()

plt.show()