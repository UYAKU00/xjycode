import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def aggregate_dataset():
    # 1. 路径配置（必须与你之前的预处理输出路径完全一致）
    PROCESSED_ROOT = r"D:\Digital_twin\step_two\data\processed_all"
    MASTER_INDEX = os.path.join(PROCESSED_ROOT, "master_index.csv")

    if not os.path.exists(MASTER_INDEX):
        print(f"❌ 找不到索引文件: {MASTER_INDEX}，请先运行预处理脚本！")
        return

    # 2. 读取总索引
    df = pd.read_csv(MASTER_INDEX)
    groups = df['group'].unique()

    # 3. 计算全局归一化基准 (V 和 Y)
    # 交叉验证需要统一的缩放标准，我们从索引中随机抽样一部分来计算基准
    print("⚖️ 正在计算全局特征归一化基准...")
    all_v, all_y, all_w = [], [], []

    # 抽取 1000 个样本计算极值即可，速度快且准确
    sample_df = df.sample(n=min(1000, len(df)), random_state=42)
    for _, row in sample_df.iterrows():
        data = np.load(row['file_path'])
        all_v.append(data['V'])
        all_y.append(data['Y'])
        all_w.append(data['W'][0])

    '''
    v_min, v_max = np.array(all_v).min(axis=0), np.array(all_v).max(axis=0)
    y_min, y_max = np.array(all_y).min(axis=0), np.array(all_y).max(axis=0)
    '''

    all_v = np.array(all_v)
    all_y = np.array(all_y)
    all_w = np.array(all_w)  # shape: (N, C, 250)

    v_min, v_max = all_v.min(axis=0), all_v.max(axis=0)
    y_min, y_max = all_y.min(axis=0), all_y.max(axis=0)

    # log-domain 的 min/max
    all_y_log = np.log(all_y + 1e-8)
    y_log_min = all_y_log.min(axis=0)
    y_log_max = all_y_log.max(axis=0)

    # 新增：每个通道一个全局均值和标准差
    w_mean = all_w.mean(axis=(0, 2), keepdims=True)  # shape: (1, C, 1)
    w_std = all_w.std(axis=(0, 2), keepdims=True)  # shape: (1, C, 1)

    # 4. 核心逻辑：为 8 个组分别生成 10 折交叉验证计划 (8:1:1)
    # cv_plan 结构: { 'Group1': [ {fold1_train:[], fold1_val:[], fold1_test:[]}, ... ], ... }
    cv_plan = {}

    print(f"🔄 正在生成 10-Fold (8:1:1) 划分方案...")

    for g in groups:
        # 筛选出当前组的 1250 个样本
        group_df = df[df['group'] == g].copy().reset_index(drop=True)
        paths = group_df['file_path'].values

        # 使用 KFold 将 1250 分成 10 份 (每份 125)
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        folds_list = []

        # 这里的 split 会循环 10 次
        indices = np.arange(len(paths))
        for fold_idx, (train_val_indices, test_indices) in enumerate(kf.split(indices)):
            # test_indices 是 125 个 (10%)
            # 在剩下的 1125 个里，再切出 125 个作为验证集 (1/9)
            val_split = len(train_val_indices) // 9
            val_indices = train_val_indices[:val_split]
            train_indices = train_val_indices[val_split:]

            folds_list.append({
                'train': paths[train_indices].tolist(),
                'val': paths[val_indices].tolist(),
                'test': paths[test_indices].tolist()
            })

        cv_plan[g] = folds_list

    # 5. 保存这个“战略地图”
    output_config = {
        'cv_plan': cv_plan,
        'norm_meta': {
            'v_min': v_min, 'v_max': v_max,
            'y_log_min': y_log_min, 'y_log_max': y_log_max,
            'w_mean': w_mean, 'w_std': w_std
        }
    }

    config_path = os.path.join(PROCESSED_ROOT, "cv_strategy.npy")
    np.save(config_path, output_config)

    print(f"✨ 数据集规划完成！")
    print("-" * 50)
    print(f"📂 策略文件: {config_path}")
    print(f"📊 覆盖分区: {len(groups)} 个")
    print(f"📈 划分标准: 每折包含 1000 训练 / 125 验证 / 125 测试")
    print(f"⚖️ HR 归一化范围: {v_min[0]:.1f} - {v_max[0]:.1f}")
    print(f"⏱️ PTT 归一化范围: {v_min[1]:.4f} - {v_max[1]:.4f}")

    print("-" * 50)


if __name__ == "__main__":
    aggregate_dataset()