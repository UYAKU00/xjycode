import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 🌟 强制使用独立窗口，防止后端冲突或白屏
matplotlib.use('TkAgg')


def plot_processed_sample():
    # ================= 1. 基础路径配置 =================
    NPZ_PATH = r"D:\Digital_twin\step_two\data\processed_all\cv_strategy.npy"

    conda_env_path = r"C:\Users\lenovo\.conda\envs\twin"
    dll_path = os.path.join(conda_env_path, "Library", "bin")
    if os.path.exists(dll_path):
        os.add_dll_directory(dll_path)
        os.environ['PATH'] = dll_path + os.pathsep + os.environ['PATH']

    if not os.path.exists(NPZ_PATH):
        print(f"❌ 找不到核心战略规划文件: {NPZ_PATH}")
        return

    # ================= 2. 加载战略地图元数据 =================
    meta = np.load(NPZ_PATH, allow_pickle=True).item()
    cv_plan = meta['cv_plan']
    norm_meta = meta['norm_meta']

    # 提取用于标量特征解耦的极值范围
    v_min = norm_meta['v_min']
    v_max = norm_meta['v_max']

    first_group = list(cv_plan.keys())[0]
    sample_path = cv_plan[first_group][0]['train'][0]

    if not os.path.exists(sample_path):
        print(f"❌ 无法定位到具体样本文件: {sample_path}")
        return

    # ================= 3. 读取单个样本并降维 =================
    print("═" * 60)
    print(f"📊 当前抽检样本绝对路径: {sample_path}")

    with np.load(sample_path, allow_pickle=True) as sample_data:
        W_raw = sample_data['W'].copy()  # 提取原始时序波形 (1, 11, 250)
        V_raw = sample_data['V'].copy()  # 提取特征 [HR, PTT]

    # 如果包含批次维度 (1, 11, 250)，强行降维成二维矩阵 (11, 250)
    if len(W_raw.shape) == 3 and W_raw.shape[0] == 1:
        W_raw = W_raw[0]

    # ================= 4. 🌟 核心：执行各通道独立的 Min-Max 归一化 =================
    # 建立一个空矩阵用于承载转化后的纯净 [0, 1] 数据
    W_norm = np.zeros_like(W_raw, dtype=np.float32)

    for i in range(W_raw.shape[0]):
        channel_min = np.nanmin(W_raw[i])
        channel_max = np.nanmax(W_raw[i])
        # 规避分母为 0 的异常，强制缩放到 [0, 1] 空间
        denom = (channel_max - channel_min) + 1e-8
        W_norm[i] = (W_raw[i] - channel_min) / denom

    # 标量特征归一化检查
    if V_raw[0] > 10.0:
        # 如果文件中存的是物理值(如3000)，利用元数据中的极值进行标准压缩
        V_norm_hr = (V_raw[0] - v_min[0]) / ((v_max[0] - v_min[0]) + 1e-8)
        V_norm_ptt = (V_raw[1] - v_min[1]) / ((v_max[1] - v_min[1]) + 1e-8)
        real_hr, real_ptt = V_raw[0], V_raw[1]
    else:
        # 如果文件里已经是归一化空间，直接读取
        V_norm_hr, V_norm_ptt = V_raw[0], V_raw[1]
        real_hr = V_norm_hr * (v_max[0] - v_min[0]) + v_min[0]
        real_ptt = V_norm_ptt * (v_max[1] - v_min[1]) + v_min[1]

    print("📈 状态报告：已成功激活【独立特征空间投影】")
    print(f"📊 变换后波形边界范围: [{np.min(W_norm):.4f} 到 {np.max(W_norm):.4f}]")
    print(f"📊 变换后特征空间坐标 -> 归一化HR: {V_norm_hr:.4f} | 归一化PTT: {V_norm_ptt:.4f}")
    print(f"✨ 对应物理溯源参考 -> 真实HR: {real_hr:.1f} bpm | 真实PTT: {real_ptt:.4f} s")
    print("═" * 60)

    # ================= 5. 高清晰度大字号绘图配置 =================
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # 页面大标题（展示归一化后的数字孪生输入表征）
    fig.suptitle(
        f'心肺数字孪生网络特征空间输入信号诊断\n Norm_HR: {V_norm_hr:.4f} | Norm_PTT: {V_norm_ptt:.4f}',
        fontsize=21, fontweight='bold', y=0.98)

    # ─── 上子图：多部位压力时序波形 (P) ───
    p_labels = ['Paop (根部主动脉压)', 'Psap (外周大动脉压)', 'Prpap (近端肺动脉压)']
    p_colors = ['#d62728', '#9467bd', '#17becf']

    for i in range(3):
        # 绘制完全归一化后的 W_norm
        ax1.plot(W_norm[i], label=p_labels[i], color=p_colors[i], linewidth=2.5)

    ax1.set_title('一、多部位压力时序波形 (P) - 各通道独立归一化空间', loc='left', fontsize=18, fontweight='bold',
                  pad=10)
    ax1.set_ylabel('归一化幅值 [0, 1]', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=14, framealpha=0.9)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 🌟 强行封锁 y 轴边界，达到你要求的完美 0-1 视觉效果
    ax1.set_ylim(-0.05, 1.05)

    # ─── 下子图：多流道容积流量波形 (Q) ───
    q_labels = [
        'Q1 (二尖瓣血流)', 'Q2 (主动脉瓣血流)', 'Q3 (右上肢解耦血流)',
        'Q4 (颈内动脉血流)', 'Q5 (左上肢血流)', 'Q6 (体循环总外周血流)',
        'Q7 (三尖瓣血流)', 'Q8 (肺动脉瓣血流)'
    ]
    q_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#e377c2', '#bcbd22', '#17becf', '#7f7f7f', '#393b79']

    for i in range(8):
        # 绘制完全归一化后的 W_norm
        ax2.plot(W_norm[i + 3], label=q_labels[i], color=q_colors[i], linewidth=2.0, alpha=0.9)

    ax2.set_title('二、多流道容积流量波形 (Q) - 各通道独立归一化空间', loc='left', fontsize=18, fontweight='bold',
                  pad=10)
    ax2.set_ylabel('归一化幅值 [0, 1]', fontweight='bold')
    ax2.set_xlabel('采样时间步长 (心动周期 250 Points)', fontsize=18, fontweight='bold')

    ax2.legend(loc='upper right', ncol=4, fontsize=11, framealpha=0.9)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # 🌟 强行封锁 y 轴边界
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_processed_sample()