import os
import json
import numpy as np
import pandas as pd
import h5py
import scipy.io as sio
from scipy.signal import find_peaks
from scipy.interpolate import interp1d


# 86-87,改位置
# 103,索引对应 p_idxs

# ================= 1. 核心算法逻辑 =================
def normalize_per_channel(W):
    """
    要求：实现各通道独立缩放 [0, 1]
    W 形状为 (9, 250)
    """
    w_min = W.min(axis=1, keepdims=True)
    w_max = W.max(axis=1, keepdims=True)


    # 防止除以 0（针对完全平直的信号）
    denom = w_max - w_min
    denom[denom == 0] = 1e-8
    return (W - w_min) / denom

def resample_cycle(t, y, length=250):
    """重采样至固定点数"""
    f = interp1d(t, y, kind='cubic', bounds_error=False, fill_value="extrapolate")
    return f(np.linspace(t[0], t[-1], length))


def calculate_ptt_internal(t, p_prox, p_dist):
    """PTT计算：利用输入波形中 Prula(Idx 0) 和 Psap(Idx 2) 的相位差"""
    # 使用梯度最大值点作为特征点锚定
    t_prox = t[np.argmax(np.gradient(p_prox, t))]
    t_dist = t[np.argmax(np.gradient(p_dist, t))]
    return max(0.001, t_dist - t_prox)

# ================= 2. 核心处理函数 =================

def process_fz_sample(sample_dir, cycle_len):
    wave_path = os.path.join(sample_dir, "waveform.mat")
    feat_path = os.path.join(sample_dir, "features.csv")
    inp_path = os.path.join(sample_dir, "inputs.json")

    # 1. 加载波形数据并提取时间向量 t
    try:
        with h5py.File(wave_path, 'r') as f:
            P, Q, t = np.array(f['P']).T, np.array(f['Q']).T, np.array(f['t']).flatten()
    except:
        mat = sio.loadmat(wave_path)
        P, Q, t = mat['P'], mat['Q'], mat['t'].flatten()


    # 2. 读取心率并转换成物理点数
    feat_df = pd.read_csv(feat_path)
    hr_value = feat_df['Hr_mean'].values[0]

    # 利用 t 计算采样率
    fs = 1.0 / (t[1] - t[0])
    # 核心：根据心率计算一个周期的固定点数长度
    points_per_cycle = int((60.0 / hr_value) * fs)

    # 3. 寻找第一个周期的起始锚点 (利用 Plv: Index 0)
    dp_dt = np.gradient(P[:, 0], t)
    # 搜索距离设为周期长度的 80%
    peaks, _ = find_peaks(dp_dt, distance=int(points_per_cycle * 0.8), prominence=np.max(dp_dt) * 0.3)


    if len(peaks) > 0:
        s = peaks[0]
        e = s + points_per_cycle  # 严格按心率时长截取
    else:
        s, e = 0, points_per_cycle

    # 越界保护
    if e > len(t):
        e = len(t)
        s = max(0, e - points_per_cycle)

    # 4. 截取时间段并重采样
    tt = t[s:e] - t[s]
    p_idxs = [4, 5, 8, 9]
    q_idxs = [0, 1, 7, 8, 10, 11, 24, 25]  # 对应指定的流量
    # q_idxs = [0, 1, 24, 25]  # 对应指定的流量-------2
    # q_idxs = [0, 24]  # 对应指定的流量--------------3
    # q_idxs = []  # 对应指定的流量-------------------4

    waves = []
    for idx in p_idxs: waves.append(resample_cycle(tt, P[s:e, idx], cycle_len))
    for idx in q_idxs: waves.append(resample_cycle(tt, Q[s:e, idx], cycle_len))

    # 5. 独立归一化处理
    W_raw = np.stack(waves, axis=0)
    W_temp = W_raw.astype(np.float32)
    time_axis = np.linspace(0, (e - s) / fs, cycle_len)
    ptt = calculate_ptt_internal(time_axis, W_temp[0, :], W_temp[1, :])

    # 去掉p_idxs 中的4 通道主动脉压力，构建真正的W
    W = W_temp[1:, :]

    # 5. 标签处理：精准剔除 Rsap(10) 和 Rvc(16)
    with open(inp_path, "r", encoding="utf-8") as f:
        inp_json = json.load(f)

    r_all = np.array(inp_json['R'])
    r_static = np.delete(r_all, [10, 16])  # 剔除后剩余 23 维
    c_static = np.array(inp_json['C'])  # 21 维
    y_label = np.concatenate([r_static, c_static]).astype(np.float32)  # 总 44 维

    v_fused = np.array([hr_value, ptt], dtype=np.float32)

    #------------------------------------------
    # ----------损失函数需要的波形提取--------------
    #------------------------------------------
    # 1:Phaa, 2:Plna, 3:Plca, 4:Paop, 5:Prula, 6:Prica, 7:Plica, 8:Plula, 9:Psap, 10:Prsv,
    # 11:Prijv, 12:Plijv, 13:Plsv, 14:Psv, 15:Pvc, 18:Prpap, 19:Plpap, 20:Prpad, 21:Plpad, 22:Prpv, 23:Plpv

    p_loss_idxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23]
    P_loss_waves = [resample_cycle(tt, P[s:e, idx], cycle_len) for idx in p_loss_idxs]


    # [需求 2] 积分、欧姆定律 需要用到的 6 个核心流量 Q (严格按照你的 allQ 顺序)
    # 1:Q2(总积分), 2:Q3(haa支流), 3:Q4(lna支流), 4:Q5(lca支流), 10:Q8(lna下游流), 11:Q9(lca下游流)
    q_loss_idxs = [1, 2, 3, 4, 10, 11]
    Q_loss_waves = [resample_cycle(tt, Q[s:e, idx], cycle_len) for idx in q_loss_idxs]

    # 堆叠成专用的物理计算矩阵 W_loss (共 27 行：前 21 行是 P，后 6 行是 Q)
    W_loss = np.stack(P_loss_waves + Q_loss_waves, axis=0).astype(np.float32)

    return W[np.newaxis, :, :], v_fused, y_label, W_loss[np.newaxis, :, :]


# ================= 3. 主程序 (适配 8 组分区遍历) =================
def main():
    # 核心修改：将路径直接指向包含 8 个 Group 的 dataset_10000 文件夹
    BASE_DIR = r"D:\Digital_twin\step_one\dataset_10000"
    # 输出路径保持不变，建议所有处理后的数据统一存放
    OUTPUT_DIR = r"D:\Digital_twin\step_two\data\processed_all"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 自动获取 dataset_10000 下的所有文件夹名（即 Group1_... 到 Group8_...）
    group_list = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    # 排序一下，保证处理顺序是 Group1, Group2...
    group_list.sort()
    index_records = []
    print(f"🚀 开始预处理，检测到目录: {BASE_DIR}")
    print(f"📊 共识别到 {len(group_list)} 个生理分区组")


    for group_name in group_list:
        group_path = os.path.join(BASE_DIR, group_name)
        print(f"📂 正在扫描分区: {group_name}...")
        processed_count = 0
        # 2. 遍历该组下的所有样本文件夹（1250个）
        for root, dirs, files in os.walk(group_path):
            # 只有当文件夹内同时包含这两个文件时才处理
            if "waveform.mat" in files and "features.csv" in files:
                sid = os.path.basename(root)  # 获取样本 ID
                try:
                    # 执行处理逻辑 (去噪、重采样、标签剔除)
                    W, V, Y, W_loss = process_fz_sample(root, 250)

                    # 统一保存：文件名格式如 Group1_Rm_Ra..._sampleID.npz
                    save_filename = f"{group_name}_{sid}.npz"
                    save_path = os.path.join(OUTPUT_DIR, save_filename)
                    np.savez_compressed(save_path, W=W, V=V, Y=Y, W_loss=W_loss)

                    # 记录索引信息
                    index_records.append({
                        "id": sid,
                        "group": group_name,
                        "file_path": save_path
                    })
                    processed_count += 1

                    # 每处理 250 个样本打印一次进度，避免刷屏
                    if processed_count % 250 == 0:
                        print(f"   已完成 {processed_count} / 1250")
                except Exception as e:
                    print(f"   ⚠️ [错误] 在 {group_name} 的样本 {sid} 处发生异常: {e}")

        print(f"   ✅ 分区 {group_name} 处理完毕，共成功转换 {processed_count} 个样本。")

    # 3. 生成全局索引表，这是后面 10 折交叉验证的“生命线”
    df_index = pd.DataFrame(index_records)
    df_index.to_csv(os.path.join(OUTPUT_DIR, "master_index.csv"), index=False)

    print(f"\n✨ 全部预处理任务圆满完成！")
    print(f"📈 最终有效样本总数: {len(df_index)}")
    print(f"📂 处理后的数据索引位于: {os.path.join(OUTPUT_DIR, 'master_index.csv')}")


if __name__ == "__main__":
    main();