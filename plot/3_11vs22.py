import matplotlib.pyplot as plt
import matplotlib

# 强制使用交互式后端，确保能弹出独立的窗口
matplotlib.use('TkAgg')

# 设置中文字体与负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================= 1. 数据配置 =================
categories = ['第一类 (Good)', '第二类 (Mid)', '第三类 (Bad)']

# 阻力参数 (R) 占比数据
r_11_channel = [26.08, 17.39, 56.52]
r_22_channel = [52.17, 17.39, 30.43]

# 顺应性参数 (C) 占比数据
c_11_channel = [4.76, 14.29, 80.95]
c_22_channel = [19.04, 19.04, 61.90]


# ================= 2. 窗口一：阻力参数 (R) 对比 =================
fig1, ax1 = plt.subplots(figsize=(10, 6))

# 绘制 11 通道与 22 通道的折线
ax1.plot(categories, r_11_channel, marker='o', linewidth=2.5, markersize=8,
         color='#ff7f0e', label='11通道输入 (Mod_11)')
ax1.plot(categories, r_22_channel, marker='s', linewidth=2.5, markersize=8,
         color='#1f77b4', label='22通道输入 (Mod_22)')

# 加上具体数值标注
for i, (v11, v22) in enumerate(zip(r_11_channel, r_22_channel)):
    ax1.text(i, v11 + 1.5, f"{v11:.2f}%", ha='center', va='bottom', color='#d35400', fontweight='bold')
    ax1.text(i, v22 - 3.0, f"{v22:.2f}%", ha='center', va='top', color='#2980b9', fontweight='bold')

# 窗口一细节精修
ax1.set_title('阻力参数 (R) 回归性能评价 —— 通道数对比', fontsize=28, fontweight='bold', pad=15)
ax1.set_ylabel('参数占比 (%)', fontsize=22)
ax1.set_ylim(0, 100)
ax1.grid(axis='y', linestyle=':', alpha=0.6)
ax1.legend(loc='upper right', fontsize=16)
ax1.tick_params(axis='x', labelsize=22)
ax1.tick_params(axis='y', labelsize=17)


# ================= 3. 窗口二：顺应性参数 (C) 对比 =================
fig2, ax2 = plt.subplots(figsize=(10, 6))

# 绘制 11 通道与 22 通道的折线
ax2.plot(categories, c_11_channel, marker='o', linewidth=2.5, markersize=8,
         color='#e377c2', label='11通道输入 (Mod_11)')
ax2.plot(categories, c_22_channel, marker='s', linewidth=2.5, markersize=8,
         color='#2ca02c', label='22通道输入 (Mod_22)')

# 加上具体数值标注
for i, (v11, v22) in enumerate(zip(c_11_channel, c_22_channel)):
    ax2.text(i, v11 + 1.5, f"{v11:.2f}%", ha='center', va='bottom', color='#c0392b', fontweight='bold')
    ax2.text(i, v22 - 3.0, f"{v22:.2f}%", ha='center', va='top', color='#27ae60', fontweight='bold')

# 窗口二细节精修
ax2.set_title('顺应性参数 (C) 回归性能评价 —— 通道数对比', fontsize=28, fontweight='bold', pad=15)
ax2.set_ylabel('参数占比 (%)', fontsize=22)
ax2.set_ylim(0, 100)
ax2.grid(axis='y', linestyle=':', alpha=0.6)
ax2.legend(loc='upper right', bbox_to_anchor=(0.92, 1.0), fontsize=16)
ax2.tick_params(axis='x', labelsize=22)
ax2.tick_params(axis='y', labelsize=17)

# ================= 4. 气泡弹出 =================
plt.show()