import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def aggregate_dataset():
    # 1. 【关键修改】路径配置：必须指向你最新的 step_two_2 目录
    PROCESSED_ROOT = r"D:\Digital_twin\step_two_2\data\processed_all"
    MASTER_INDEX = os.path.join(PROCESSED_ROOT, "master_index.csv")

    if not os.path.exists(MASTER_INDEX):
        print(f"❌ 找不到索引文件: {MASTER_INDEX}，请先运行 preprocess_cycles.py！")
        return

    # 2. 读取总索引
    df = pd.read_csv(MASTER_INDEX)
    groups = df['group'].unique()

    # 3. 计算全局归一化基准 (V 和 Y)
    # 因为 V 包含了 PTT，不同分区的 PTT 差异可能很大，所以我们要扫描全量数据获取极值
    print("⚖️ 正在计算全分区特征归一化基准 (HR, PTT, R, C)...")
    all_v, all_y, all_w = [], [], []

    # 为了严谨，建议扫描所有样本（一万个样本扫描很快，约 10 秒）
    for _, row in df.iterrows():
        data = np.load(row['file_path'])
        all_v.append(data['V'])  # [HR, PTT]
        all_y.append(data['Y'])  # [44维参数]
        all_w.append(data['W'][0])  # 波形原参数，为后续计算平均值

    # 转换为 numpy 计算全局极值
    all_v_np = np.array(all_v)
    all_y_np = np.array(all_y)
    all_w_np = np.array(all_w)  # 形状: (N, 22, 250)

    v_min, v_max = all_v_np.min(axis=0), all_v_np.max(axis=0)
    y_min, y_max = all_y_np.min(axis=0), all_y_np.max(axis=0)

    # log-domain 的 min/max
    all_y_log = np.log(all_y_np + 1e-8)
    y_log_min = all_y_log.min(axis=0)
    y_log_max = all_y_log.max(axis=0)

    w_mean = all_w_np.mean(axis=(0, 2), keepdims=True)
    w_std = all_w_np.std(axis=(0, 2), keepdims=True)


    # 4. 生成 8:1:1 滚动划分方案
    cv_plan = {}
    print(f"🔄 正在为 {len(groups)} 个分区生成 10-Fold 滚动方案...")

    for g in groups:
        group_df = df[df['group'] == g].copy().reset_index(drop=True)
        paths = group_df['file_path'].values

        # 使用 KFold 将该组的 1250 个样本分成 10 份
        # shuffle=True 保证了数据在每一块里的随机性，random_state 保证了实验可复现
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        folds_list = []

        indices = np.arange(len(paths))
        for fold_idx, (train_val_indices, test_indices) in enumerate(kf.split(indices)):
            # 这里的 train_val_indices 有 1125 个
            # 我们从中再切出 125 个作为“二次训练集 (Val)”
            val_split_size = 125
            val_indices = train_val_indices[:val_split_size]
            train_indices = train_val_indices[val_split_size:]

            # 最终比例：1000 (Train) : 125 (Val) : 125 (Test)
            folds_list.append({
                'train': paths[train_indices].tolist(),
                'val': paths[val_indices].tolist(),
                'test': paths[test_indices].tolist()
            })

        cv_plan[g] = folds_list

    # 5. 保存“战略地图”
    output_config = {
        'cv_plan': cv_plan,
        'norm_meta': {
            'v_min': v_min, 'v_max': v_max,
            'y_min': y_min, 'y_max': y_max,
            'y_log_min': y_log_min, 'y_log_max': y_log_max,
            'w_mean': w_mean, 'w_std': w_std
        }
    }

    config_path = os.path.join(PROCESSED_ROOT, "cv_strategy.npy")
    np.save(config_path, output_config)

    print(f"✨ [战略地图] 生成成功！")
    print("-" * 60)
    print(f"📂 存放位置: {config_path}")
    print(f"⚖️ 全局 HR 范围: {v_min[0]:.1f} - {v_max[0]:.1f} bpm")
    print(f"⚖️ 全局 PTT 范围: {v_min[1]:.4f} - {v_max[1]:.4f} s")
    print(f"📊 划分结果: 每轮集结 8000(Source) + 1000(Val) | 1000(隔离测试)")
    print("-" * 60)


if __name__ == "__main__":
    aggregate_dataset()