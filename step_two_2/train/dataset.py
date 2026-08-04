import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ================= 1. 自定义数据集类 =================
class PhysioDataset(Dataset):
    """
    生理信号数据集：动态加载 .npz 文件 (22通道波形 + 2维特征 + 44维标签 + 27通道物理波形)
    """

    def __init__(self, file_paths, norm_meta, is_train=False):
        self.file_paths = file_paths
        self.v_min = torch.tensor(norm_meta['v_min'], dtype=torch.float32)
        self.v_max = torch.tensor(norm_meta['v_max'], dtype=torch.float32)
        self.y_log_min = torch.tensor(norm_meta['y_log_min'], dtype=torch.float32)
        self.y_log_max = torch.tensor(norm_meta['y_log_max'], dtype=torch.float32)
        self.w_mean = torch.tensor(norm_meta['w_mean'], dtype=torch.float32)
        self.w_std = torch.tensor(norm_meta['w_std'], dtype=torch.float32)
        self.is_train = is_train

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])
        w = torch.from_numpy(data['W']).float()  # (1, 22, 250)

        # 全局 Z-Score 标准化
        w = (w - self.w_mean) / (self.w_std + 1e-8)
        w = torch.clamp(w, -10, 10)

        # 特征 V 处理 (HR, PTT)
        v = torch.from_numpy(data['V']).float()
        v = (v - self.v_min) / (self.v_max - self.v_min + 1e-8)

        # 标签 Y 处理 (44维参数) - 执行 Log 域归一化
        y = torch.from_numpy(data['Y']).float()
        y = torch.log(y + 1e-8)
        y = (y - self.y_log_min) / (self.y_log_max - self.y_log_min + 1e-8)

        # 提取物理损失计算用波形 (27, 250)
        w_loss = torch.tensor(data['W_loss'][0], dtype=torch.float32)

        # 动态注入噪声 (仅限训练相关的 pretrain/finetune 阶段)
        if self.is_train:
            noise = torch.randn_like(w) * 0.01
            w = w + noise

        return w.squeeze(0), v, y, w_loss


# ================= 2. 调度函数：获取 4 阶段数据加载器 =================

def get_fold_loaders(group_name, fold_idx, batch_size=64):
    """
    实现 8:2:1:1 架构的数据分发
    返回: pre_loader, fine_loader, val_loader, test_loader, norm_meta
    """
    STRATEGY_PATH = r"D:\Digital_twin\step_two_2\data\processed_all\cv_strategy.npy"

    if not os.path.exists(STRATEGY_PATH):
        raise FileNotFoundError("❌ 找不到战略地图！请先运行 aggregate_dataset.py")

    meta = np.load(STRATEGY_PATH, allow_pickle=True).item()
    cv_plan = meta['cv_plan']
    norm_meta = meta['norm_meta']

    if group_name not in cv_plan:
        raise ValueError(f"❌ 组名 {group_name} 不在计划中")

    plan = cv_plan[group_name][fold_idx]

    # 🌟 核心划分逻辑：将原训练池 8:2 拆分为预训练和微调
    full_train_paths = plan['train']
    split_point = int(len(full_train_paths) * 0.8)

    pretrain_paths = full_train_paths[:split_point]
    finetune_paths = full_train_paths[split_point:]

    # 构建 4 个互斥数据集
    pre_ds = PhysioDataset(pretrain_paths, norm_meta, is_train=True)
    fine_ds = PhysioDataset(finetune_paths, norm_meta, is_train=True)  # 微调阶段也建议保持噪声增强
    val_ds = PhysioDataset(plan['val'], norm_meta, is_train=False)  # 裁判集
    test_ds = PhysioDataset(plan['test'], norm_meta, is_train=False)  # 盲考集

    # 构建加载器
    pre_loader = DataLoader(pre_ds, batch_size=batch_size, shuffle=True)
    fine_loader = DataLoader(fine_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return pre_loader, fine_loader, val_loader, test_loader, norm_meta


# ================= 3. 测试脚本 =================

if __name__ == "__main__":
    try:
        # 修改为你的实际组名进行测试
        test_group = "Group1_Rm_Ra_Raop_Caop_Csap"

        loaders = get_fold_loaders(test_group, fold_idx=0, batch_size=32)
        pre_l, fine_l, val_l, test_l, meta = loaders

        # 验证张量形状
        w, v, y, w_loss = next(iter(pre_l))
        print(f"\n--- 张量结构验证 ---")
        print(f"   Input Waveform: {w.shape} (22 channels)")
        print(f"   Input Features: {v.shape} (HR, PTT)")
        print(f"   Target Labels : {y.shape} (44 params)")
        print(f"   Physic Waves  : {w_loss.shape} (27 channels)")

    except Exception as e:
        print(f"❌ 测试失败: {e}")