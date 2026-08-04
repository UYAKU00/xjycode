import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ================= 1. 自定义数据集类 =================
class PhysioDataset(Dataset):
    """
    生理信号数据集：根据路径列表动态加载单个 .npz 文件
    """

    def __init__(self, file_paths, norm_meta, is_train=False):
        self.file_paths = file_paths
        self.v_min = torch.tensor(norm_meta['v_min'], dtype=torch.float32)
        self.v_max = torch.tensor(norm_meta['v_max'], dtype=torch.float32)
        # self.y_min = torch.tensor(norm_meta['y_min'], dtype=torch.float32)
        # self.y_max = torch.tensor(norm_meta['y_max'], dtype=torch.float32)
        self.y_log_min = torch.tensor(norm_meta['y_log_min'], dtype=torch.float32)
        self.y_log_max = torch.tensor(norm_meta['y_log_max'], dtype=torch.float32)
        self.w_mean = torch.tensor(norm_meta['w_mean'], dtype=torch.float32)
        self.w_std = torch.tensor(norm_meta['w_std'], dtype=torch.float32)

        self.is_train = is_train

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # 1. 加载数据
        data = np.load(self.file_paths[idx])
        w = torch.from_numpy(data['W']).float()  # (1, N, 250)
        w = (w - self.w_mean) / (self.w_std + 1e-8)
        w = torch.clamp(w, -10, 10)

        v = torch.from_numpy(data['V']).float()  # (2,)
        y = torch.from_numpy(data['Y']).float()  # (44,)
        w_loss = torch.tensor(data['W_loss'][0], dtype=torch.float32)

        # 2. 特征 V 和 标签 Y 的实时归一化 (使用全局基准)
        v = (v - self.v_min) / (self.v_max - self.v_min + 1e-8)
        # y = (y - self.y_min) / (self.y_max - self.y_min + 1e-8) # 如果标签也需要归一化可开启

        y = torch.from_numpy(data['Y']).float()  # (44,)
        if torch.any(y <= 0):
            raise ValueError(f"Y contains non-positive values in file: {self.file_paths[idx]}")

        y = torch.log(y + 1e-8)
        y = ( y - self.y_log_min) / ( self.y_log_max - self.y_log_min + 1e-8)


        # 3. 动态注入噪声 (仅限训练集，0.01强度)
        if self.is_train:
            noise = torch.randn_like(w) * 0.01
            w = w + noise

        # 去掉 w 多余的 Batch 维度，变成 (11, 250)
        return w.squeeze(0), v, y, w_loss


# ================= 2. 调度函数：获取指定折的数据加载器 =================

def get_fold_loaders(group_name, fold_idx, batch_size=64):
    """
    根据战略地图，获取特定分区、特定一折的加载器
    """
    # 路径对接
    STRATEGY_PATH = r"D:\Digital_twin\step_two\data\processed_all\cv_strategy.npy"

    if not os.path.exists(STRATEGY_PATH):
        raise FileNotFoundError("❌ 找不到战略地图！请先运行 aggregate_dataset.py")

    # 1. 加载地图
    meta = np.load(STRATEGY_PATH, allow_pickle=True).item()
    cv_plan = meta['cv_plan']
    norm_meta = meta['norm_meta']

    # 2. 提取指定分区和折数的路径
    if group_name not in cv_plan:
        raise ValueError(f"❌ 组名 {group_name} 不在计划中")

    plan = cv_plan[group_name][fold_idx]

    # 3. 构建 Dataset
    train_ds = PhysioDataset(plan['train'], norm_meta, is_train=True)
    val_ds = PhysioDataset(plan['val'], norm_meta, is_train=False)
    test_ds = PhysioDataset(plan['test'], norm_meta, is_train=False)

    # 4. 创建 DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, norm_meta


if __name__ == "__main__":
    # 测试一下：获取 Group1 的第 0 折数据
    try:
        tl, vl, tsl, meta = get_fold_loaders("Group1_Rm_Ra_Raop_Caop_Csap", 0, batch_size=32)
        w, v, y, w_loss = next(iter(tl))
        print(f"✅ 10折加载测试成功！")
        print(f"   Waveform: {w.shape} (应为 32, N, 250)")
        print(f"   损失函数(w_loss): {w_loss.shape} (应为 32, 27, 250)")
        print(f"   Features: {v.shape} (应为 32, 2)")
        print(f"   Labels: {y.shape} (应为 32, 44)")
    except Exception as e:
        print(f"❌ 测试失败: {e}")