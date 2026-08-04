import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 强制使用独立窗口，防止 PyCharm 内部白屏
matplotlib.use('TkAgg')

def plot_processed_sample():
    # 1. 【修改点】路径检查：指向最新的 step_two_2 打包文件
    NPZ_PATH = r"D:\Digital_twin\step_two_2\data\processed\train_final.npz"

    if not os.path.exists(NPZ_PATH):
        print(f"❌ 找不到文件: {NPZ_PATH}")
        return

    # 2. 加载数据
    print("🔍 正在加载数据进行诊断...")
    data = np.load(NPZ_PATH)

    # 【重要检查】确认 data_aggregate.py 是否保存了极值
    if 'v_min' not in data:
        print("❌ 错误：检测到旧版 Z-Score 数据。请先运行 data_aggregate.py 生成 Min-Max 格式数据。")
        return

    # 【修改点】更新注释，现在是 17 通道
    W = data['train_W']   # 波形 (N, 17, 250)
    V = data['train_V']   # 归一化特征 (N, 2)
    Y = data['train_Y']   # 标签 (N, 44)
    v_min = data['v_min'] # [min_hr, min_ptt]
    v_max = data['v_max'] # [max_hr, max_ptt]

    # --- 核心诊断输出 ---
    print("-" * 50)
    print(f"📊 波形矩阵形状 (W): {W.shape}")
    print(f"📊 特征矩阵形状 (V): {V.shape}")
    print(f"📊 标签维度: {Y.shape[1]} (应为 44)")

    # 检查波形数值是否都在 0-1 之间
    max_w = np.nanmax(W)
    min_w = np.nanmin(W)
    print(f"📈 波形归一化范围: [{min_w:.4f} 到 {max_w:.4f}]")
    print("-" * 50)

    # 3. 随机抽取并还原物理数值
    idx = np.random.randint(0, len(W))
    sample_w = W[idx]

    # 根据 Min-Max 逻辑还原物理数值
    real_hr = V[idx][0] * (v_max[0] - v_min[0]) + v_min[0]
    real_ptt = V[idx][1] * (v_max[1] - v_min[1]) + v_min[1]

    print(f"✨ 正在展示索引为 {idx} 的周期数据...")
    print(f"📊 物理特征还原 -> 心率: {real_hr:.1f} bpm, PTT: {real_ptt:.4f} s")

    # 4. 绘图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 【修改点】创建 3 个子图，并增加画板高度 (figsize)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'数字孪生输入信号诊断 (索引: {idx})\n还原物理值 -> HR: {real_hr:.1f} | PTT: {real_ptt:.4f}', fontsize=12)

    # ================= 第一张图 =================
    # 压力波形 (P: 前4路, 对应 sample_w 的索引 0~3)
    p_labels = ['P_idx4', 'P_idx5(Prula)', 'P_idx8(Plula)', 'P_idx9(Psap)']
    p_colors = ['#1f77b4', '#d62728', '#ff7f0e', '#9467bd']
    for i in range(4):
        ax1.plot(sample_w[i], label=p_labels[i], color=p_colors[i], linewidth=1.5)

    ax1.set_title('压力波形 (P) - 4通道', loc='left', fontweight='bold')
    ax1.set_ylabel('幅值 [0, 1]')
    ax1.legend(loc='upper right', ncol=4, fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.5)

    # ================= 第二张图 =================
    # 流量波形 (Q: 中间9路, 对应 sample_w 的索引 4~12)
    q_labels = ['Q0(二尖瓣)', 'Q1(主瓣)', 'Q10(左脑)', 'Q11(左臂)', 'Q12', 'Q24(三尖瓣)', 'Q25(肺瓣)', 'Q26', 'Q27']
    for i in range(9):
        ax2.plot(sample_w[i + 4], label=q_labels[i], alpha=0.8)

    ax2.set_title('流量波形 (Q) - 9通道', loc='left', fontweight='bold')
    ax2.set_ylabel('幅值 [0, 1]')
    ax2.legend(loc='upper right', ncol=5, fontsize=8)
    ax2.grid(True, linestyle=':', alpha=0.5)

    # ================= 第三张图 =================
    # 容积波形 (V: 最后4路, 对应 sample_w 的索引 13~16)
    v_labels = ['V_idx0', 'V_idx16', 'V_idx17', 'V_idx24']
    for i in range(4):
        ax3.plot(sample_w[i + 13], label=v_labels[i], alpha=0.8, linewidth=1.5)

    ax3.set_title('容积/其他波形 (V) - 4通道', loc='left', fontweight='bold')
    ax3.set_ylabel('幅值 [0, 1]')
    ax3.set_xlabel('采样点 (心动周期 250 pts)')
    ax3.legend(loc='upper right', ncol=4, fontsize=9)
    ax3.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_processed_sample()