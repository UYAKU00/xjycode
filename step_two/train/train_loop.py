import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# 🌟 强行将 conda 环境的 Library\bin 加入系统搜索路径
conda_env_path = r"C:\Users\lenovo\.conda\envs\twin"
dll_path = os.path.join(conda_env_path, "Library", "bin")
if os.path.exists(dll_path):
    os.add_dll_directory(dll_path)
    os.environ['PATH'] = dll_path + os.pathsep + os.environ['PATH']

# 注意：根据你的实际文件名修改导入
from model_def import TwinFusionNet
from dataset_loader import PhysioDataset, DataLoader


# ================= 1. 验证与可视化函数 (全局) =================

def run_evaluation(model, loader, criterion, device):
    """裁判员函数：在验证集上评估模型性能"""
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for w, v, y, w_loss in loader:
            w, v, y, w_loss = w.to(device), v.to(device), y.to(device), w_loss.to(device)
            preds = model(w, v)
            loss, _ = criterion(preds, y, w_loss, phys_weight=0.5)
            total_val_loss += loss.item() * w.size(0)
    return total_val_loss / len(loader.dataset)


def plot_training_history(history, save_path):
    """自动生成收敛曲线图"""
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 6))
    plt.semilogy(epochs, history["train_loss"], label='Train Loss (SmoothL1)', color='#1f77b4')
    plt.semilogy(epochs, history["val_loss"], label='Val Loss (Monitor)', color='#ff7f0e')
    if len(epochs) > 50:
        plt.axvline(x=50, color='gray', linestyle='--', label='Stage B Start')
    plt.xlabel('Total Epochs')
    plt.ylabel('Loss Value (Log Scale)')
    plt.title('11-Channel Model Training Convergence')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(save_path)
    plt.close()


# ================= 2. 基础配置 =================
ROOT_DIR = r"D:\Digital_twin\step_two"
MOD_DIR = os.path.join(ROOT_DIR, "mod_1_11_no")
STRATEGY_PATH = os.path.join(ROOT_DIR, r"data\processed_all\cv_strategy.npy")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS_PRE = 50
EPOCHS_FINE = 20


# ================= 3. 核心：物理损失函数 =================
class PhysicsInformedLoss(nn.Module):
    def __init__(self, y_log_min, y_log_max, dt=0.005, lambdas=(1.0, 0, 0.005, 0.001, 0.002, 0.001)):
        super(PhysicsInformedLoss, self).__init__()
        self.dt = dt
        self.y_log_min = y_log_min
        self.y_log_max = y_log_max

        # 损失项比重：数据、方向、频域、积分、肺部衰减、容积守恒
        self.l_data, self.l_cos, self.l_fft, self.l_int, self.l_pulm, self.l_mass = lambdas
        self.l_decay = 0.001  # 系统循环衰减系数

        # 11通道参数权重 (保持44维，与模型输出对齐)
        r_weights = [1.2, 0.6, 1.3, 1.3, 1.3, 1.5, 1.2, 1.1, 0.6, 1.1, 0.8, 0.8, 0.6, 0.7, 1.3, 1.3, 0.6, 0.6, 0.6, 0.6,
                     0.6, 0.8, 0.8]
        c_weights = [1.3, 1.0, 1.0, 1.3, 0.7, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.8,
                     0.8]
        self.register_buffer("param_weights", torch.tensor(r_weights + c_weights, dtype=torch.float32))

        # 🌟 核心：组合参数监督索引 (用于 44 维输出的逻辑对齐)
        self.register_buffer("idx_R_pulm_comb", torch.tensor([16, 17, 18, 19, 20], dtype=torch.long))
        self.register_buffer("idx_C_venous_comb", torch.tensor([32, 33, 34, 35, 36], dtype=torch.long))

    def forward(self, preds, targets, w_loss, phys_weight=1.0):
        # --- 1. 基础统计损失 (数据 + 方向 + 频域) ---
        mse_per_param = (preds - targets) ** 2
        smooth_per_param = F.smooth_l1_loss(preds, targets, reduction='none')
        weighted_loss = (0.7 * mse_per_param + 0.3 * smooth_per_param) * self.param_weights.view(1, -1)
        loss_data = weighted_loss.mean() / (self.param_weights.mean() + 1e-8)

        cos_sim = F.cosine_similarity(preds, targets, dim=1)
        loss_cos = torch.mean(1.0 - cos_sim)

        fft_preds = torch.fft.rfft(preds, dim=1)
        fft_targets = torch.fft.rfft(targets, dim=1)
        loss_fft = F.mse_loss(fft_preds.abs(), fft_targets.abs())

        # --- 2. 物理量还原 (Log 域反转) ---
        preds_log = preds * (self.y_log_max - self.y_log_min + 1e-8) + self.y_log_min
        preds_phys = torch.exp(preds_log)
        targets_phys = torch.exp(targets * (self.y_log_max - self.y_log_min + 1e-8) + self.y_log_min)

        # 🌟 3. 组合参数等效监督 (防止单个参数乱飘)
        def calc_comb_loss(p_phys, t_phys, idxs):
            p_sum = torch.sum(p_phys[:, idxs], dim=1)
            t_sum = torch.sum(t_phys[:, idxs], dim=1)
            return torch.mean(((p_sum - t_sum) / (t_sum + 1e-6)) ** 2)

        loss_comb = calc_comb_loss(preds_phys, targets_phys, self.idx_R_pulm_comb) + \
                    calc_comb_loss(preds_phys, targets_phys, self.idx_C_venous_comb)

        # --- 4. 宏观生理机理 PDE 约束 ---
        R_pred, C_pred = preds_phys[:, 0:23], preds_phys[:, 23:44]

        # 11通道关键波形索引 (请确认 w_loss 的顺序)
        idx_P_aop, idx_P_sap, idx_P_rpap, idx_Q2 = 3, 8, 15, 21
        Paop, Psap, Prpap = w_loss[:, idx_P_aop, :], w_loss[:, idx_P_sap, :], w_loss[:, idx_P_rpap, :]
        Q2 = w_loss[:, idx_Q2, :]

        # [积分守恒] 顺应性与搏出量匹配
        SV = torch.trapz(Q2, dx=self.dt, dim=1)
        PP = torch.max(Paop, dim=1)[0] - torch.min(Paop, dim=1)[0]
        C_art_sum = torch.sum(C_pred[:, :9], dim=1)
        target_C = SV / (PP + 1e-6)
        loss_integral = torch.mean(((C_art_sum - target_C) / (target_C + 1.0)) ** 2)

        # [衰减约束] 系统与肺循环压力下降逻辑
        Q2_max = torch.max(Q2, dim=1, keepdim=True)[0]
        diastole_mask = (Q2 < 0.05 * Q2_max).float()

        # 系统循环衰减
        dPsap_dt = torch.gradient(Psap, spacing=(self.dt,), dim=1)[0]
        TPR_eq = torch.sum(R_pred[:, 2:15], dim=1).unsqueeze(1)
        decay_res_sys = (dPsap_dt + Psap / (TPR_eq * C_art_sum.unsqueeze(1) + 1e-6)) / (Psap + 1.0)
        loss_decay = torch.sum((decay_res_sys * diastole_mask) ** 2) / (torch.sum(diastole_mask) + 1e-6)

        # 肺循环衰减
        dPrpap_dt = torch.gradient(Prpap, spacing=(self.dt,), dim=1)[0]
        R_pulm_eq = torch.sum(R_pred[:, 17:23], dim=1).unsqueeze(1)
        C_pulm_sum = torch.sum(C_pred[:, 15:19], dim=1).unsqueeze(1)
        decay_res_pulm = (dPrpap_dt + Prpap / (R_pulm_eq * C_pulm_sum + 1e-6)) / (Prpap + 1.0)
        loss_pulm_decay = torch.sum((decay_res_pulm * diastole_mask) ** 2) / (torch.sum(diastole_mask) + 1e-6)

        # [容积守恒] 周期闭合
        P_all = w_loss[:, 0:21, :]  # 假设 w_loss 包含各腔室压力
        V_start = torch.sum(C_pred * P_all[:, :, 0], dim=1)
        V_end = torch.sum(C_pred * P_all[:, :, -1], dim=1)
        loss_mass = torch.mean(((V_start - V_end) / (V_start + 1.0)) ** 2)

        # --- 5. 汇总 ---
        stat_loss = (self.l_data * loss_data) + (self.l_cos * loss_cos) + (self.l_fft * loss_fft)
        phys_loss = (self.l_int * loss_integral) + (self.l_decay * loss_decay) + \
                    (self.l_pulm * loss_pulm_decay) + (self.l_mass * loss_mass)

        # 这里的 0.05 是对组合监督的加权，phys_weight 控制 PDE 项
        total_loss = stat_loss + (0.05 * loss_comb) + phys_weight * phys_loss

        return total_loss, loss_data



# ================= 4. 主训练函数 =================
def train_loop():
    if not os.path.exists(STRATEGY_PATH): return print("❌ 找不到战略地图！")
    meta = np.load(STRATEGY_PATH, allow_pickle=True).item()
    cv_plan, norm_meta = meta['cv_plan'], meta['norm_meta']

    y_log_min_t = torch.tensor(norm_meta['y_log_min'], dtype=torch.float32).to(DEVICE)
    y_log_max_t = torch.tensor(norm_meta['y_log_max'], dtype=torch.float32).to(DEVICE)

    criterion = PhysicsInformedLoss(y_log_min_t, y_log_max_t).to(DEVICE)
    test_registry = {}
    registry_path = os.path.join(MOD_DIR, "test_registry.json")

    for t_idx in range(1, 11):
        train_folder = os.path.join(MOD_DIR, f"train_{t_idx}")
        os.makedirs(train_folder, exist_ok=True)
        print_progress_map(t_idx)

        # 🌟 数据动态集结 (8:2 拆分)
        all_pre_paths, all_fine_paths, all_val_paths, all_test_paths = [], [], [], []
        for g_name in cv_plan.keys():
            plan = cv_plan[g_name][t_idx - 1]
            full_train = plan['train']
            split_point = int(len(full_train) * 0.8)
            all_pre_paths.extend(full_train[:split_point])
            all_fine_paths.extend(full_train[split_point:])
            all_val_paths.extend(plan['val'])
            all_test_paths.extend(plan['test'])

        pre_loader = DataLoader(PhysioDataset(all_pre_paths, norm_meta, is_train=True), batch_size=BATCH_SIZE,
                                shuffle=True)
        fine_loader = DataLoader(PhysioDataset(all_fine_paths, norm_meta, is_train=True), batch_size=BATCH_SIZE,
                                 shuffle=True)
        val_loader = DataLoader(PhysioDataset(all_val_paths, norm_meta, is_train=False), batch_size=BATCH_SIZE,
                                shuffle=False)

        model = TwinFusionNet(n_channels=11, n_features=2, n_outputs=44).to(DEVICE)
        history = {"train_loss": [], "val_loss": [], "phys_weight": []}
        best_val_loss = float('inf')

        # --- Stage A: 预训练 ---
        print(f"\n▶️ [Stage A] 11通道混合分区预训练 (样本: {len(all_pre_paths)})")
        optimizer = optim.Adam(model.parameters(), lr=0.0003)
        for epoch in range(1, EPOCHS_PRE + 1):
            current_phys_weight = min(0.5, max(0.0, (epoch - 30) / 20.0))
            model.train()
            pure_data_loss_sum = 0.0
            for w, v, y, w_loss in tqdm(pre_loader, desc=f"Pre {epoch}", leave=False):
                w, v, y, w_loss = w.to(DEVICE), v.to(DEVICE), y.to(DEVICE), w_loss.to(DEVICE)
                optimizer.zero_grad()
                preds = model(w, v)
                loss, l_data = criterion(preds, y, w_loss, phys_weight=current_phys_weight)
                loss.backward();
                optimizer.step()
                pure_data_loss_sum += l_data.item() * w.size(0)

            avg_data_loss = pure_data_loss_sum / len(pre_loader.dataset)
            val_score = run_evaluation(model, val_loader, criterion, DEVICE)
            history["train_loss"].append(avg_data_loss);
            history["val_loss"].append(val_score);
            history["phys_weight"].append(current_phys_weight)
            if val_score < best_val_loss:
                best_val_loss = val_score
                torch.save(model.state_dict(), os.path.join(train_folder, "best_source_model.pth"))
            print(f" Epoch {epoch:2d} | Val Loss: {val_score:.6f}")

        # --- Stage B: 微调 ---
        print(f"\n▶️ [Stage B] 11通道微调 (样本: {len(all_fine_paths)})")
        model.load_state_dict(torch.load(os.path.join(train_folder, "best_source_model.pth")))
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        for epoch in range(1, EPOCHS_FINE + 1):
            model.train()
            pure_data_loss_sum = 0.0
            for w, v, y, w_loss in tqdm(fine_loader, desc=f"Fine {epoch}", leave=False):
                w, v, y, w_loss = w.to(DEVICE), v.to(DEVICE), y.to(DEVICE), w_loss.to(DEVICE)
                optimizer.zero_grad()
                preds = model(w, v)
                loss, l_data = criterion(preds, y, w_loss, phys_weight=0.5)
                loss.backward();
                optimizer.step()
                pure_data_loss_sum += l_data.item() * w.size(0)

            avg_data_loss = pure_data_loss_sum / len(fine_loader.dataset)
            val_score = run_evaluation(model, val_loader, criterion, DEVICE)
            history["train_loss"].append(avg_data_loss);
            history["val_loss"].append(val_score);
            history["phys_weight"].append(0.5)
            if val_score < best_val_loss:
                best_val_loss = val_score
                torch.save(model.state_dict(), os.path.join(train_folder, "val_model.pth"))
            print(f" Fine Epoch {epoch:2d} | Val Loss: {val_score:.6f}")

        # 归一化归档
        with open(os.path.join(train_folder, "history.json"), "w") as f:
            json.dump(history, f)
        plot_training_history(history, os.path.join(train_folder, "loss_curve.png"))
        test_registry[f"train_{t_idx}"] = all_test_paths
        with open(registry_path, "w") as f:
            json.dump(test_registry, f)


def print_progress_map(t_idx):
    status = [" [X] " if i == (t_idx + 8) % 10 else " [V] " if i == (t_idx + 7) % 10 else "  T  " for i in range(10)]
    print(f"\n" + "═" * 70 + f"\n📊 11-Channel Fold {t_idx}\n🗺️ {''.join(status)}\n" + "═" * 70)


if __name__ == "__main__":
    os.makedirs(MOD_DIR, exist_ok=True)
    train_loop()