import os
import sys
# 🌟 强行将 conda 环境的 Library\bin 加入系统搜索路径
# 确保 PyTorch 启动时能一眼看到 nvrtc-builtins64_121.dll
conda_env_path = r"C:\Users\lenovo\.conda\envs\twin"
dll_path = os.path.join(conda_env_path, "Library", "bin")

if os.path.exists(dll_path):
    os.add_dll_directory(dll_path)
    # 同时加入系统环境变量
    os.environ['PATH'] = dll_path + os.pathsep + os.environ['PATH']

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
from tqdm import tqdm


# -----145损失函数比重--------
# -----171改通道数----------

# ================= 1. 环境与路径强制配置 (锁定 22 通道) =================
PROJECT_ROOT = r"D:\Digital_twin\step_two_2"
MOD_DIR = os.path.join(PROJECT_ROOT, "mod_2_22_no")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "processed_all")
STRATEGY_PATH = os.path.join(DATA_ROOT, "cv_strategy.npy")
TRAIN_DIR = os.path.join(PROJECT_ROOT, "train")

if TRAIN_DIR not in sys.path:
    sys.path.insert(0, TRAIN_DIR)

from model import TwinFusionNet
from dataset import PhysioDataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS_PRE = 50
EPOCHS_FINE = 20


# ================= 2. 核心：稳健版物理损失函数 (PINN Loss V2 + 余弦相似度) =================
class PhysicsInformedLoss(nn.Module):
    def __init__(self, y_min, y_max, dt=0.005, lambdas=(1.0, 1.0, 0.01, 0.001, 0.001, 0.001)):
        super(PhysicsInformedLoss, self).__init__()
        self.data_loss_fn = nn.SmoothL1Loss()
        self.dt = dt
        self.y_min = y_min
        self.y_max = y_max

        self.l_data, self.l_cos, self.l_fft, self.l_int, self.l_pulm, self.l_mass = lambdas
        self.l_decay = 0.001

    def forward(self, preds, targets, w_loss, phys_weight=1.0):

        # --- 1. 基础数据损失与样本加权 (解决垂直条带的关键) ---
        loss_data = self.data_loss_fn(preds, targets)


        # --- 2. 方向损失 ---
        # 余弦相似度损失
        cos_sim = F.cosine_similarity(preds, targets, dim=1)
        loss_cos = torch.mean(1.0 - cos_sim)


        # ------- 3. 频域损失 --------
        # 将 44 个参数视为序列进行快速傅里叶变换
        # 强制模型学习参数序列的“起伏特征”，防止预测结果平庸化
        fft_preds = torch.fft.rfft(preds, dim=1)
        fft_targets = torch.fft.rfft(targets, dim=1)
        loss_fft = F.mse_loss(fft_preds.abs(), fft_targets.abs())


        # ========= 物理约束部分 ============
        preds_phys = preds * (self.y_max - self.y_min + 1e-8) + self.y_min
        # 拆分物理量
        R_pred = preds_phys[:, 0:23]
        C_pred = preds_phys[:, 23:44]

        # --- 宏观物理约束计算 (保持你原有逻辑) ---
        idx_C_sys_art = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        idx_C_pulm_art = [15, 16, 17, 18]
        idx_R_sys_group = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        idx_R_pulm_group = [17, 18, 19, 20, 21, 22]
        idx_P_aop, idx_P_sap, idx_P_rpap = 3, 8, 15
        idx_Q2 = 21

        Paop, Psap, Prpap = w_loss[:, idx_P_aop, :], w_loss[:, idx_P_sap, :], w_loss[:, idx_P_rpap, :]
        Q2 = w_loss[:, idx_Q2, :]

        # [Loss 4] 积分守恒
        SV = torch.trapz(Q2, dx=self.dt, dim=1)
        PP = torch.max(Paop, dim=1)[0] - torch.min(Paop, dim=1)[0]
        C_art_sum = torch.sum(C_pred[:, idx_C_sys_art], dim=1)
        target_C = SV / (PP + 1e-6)
        loss_integral = torch.mean(((C_art_sum - target_C) / (target_C + 1.0)) ** 2)

        # [Loss 5 ] 衰减约束
        Q2_max = torch.max(Q2, dim=1, keepdim=True)[0]
        diastole_mask = (Q2 < 0.05 * Q2_max).float()
        dPsap_dt = torch.gradient(Psap, spacing=(self.dt,), dim=1)[0]
        TPR_eq = torch.sum(R_pred[:, idx_R_sys_group], dim=1).unsqueeze(1)
        decay_res_sys = (dPsap_dt + Psap / (TPR_eq * C_art_sum.unsqueeze(1) + 1e-6)) / (Psap + 1.0)
        loss_decay = torch.sum((decay_res_sys * diastole_mask) ** 2) / (torch.sum(diastole_mask) + 1e-6)

        dPrpap_dt = torch.gradient(Prpap, spacing=(self.dt,), dim=1)[0]
        R_pulm_eq = torch.sum(R_pred[:, idx_R_pulm_group], dim=1).unsqueeze(1)
        C_pulm_sum = torch.sum(C_pred[:, idx_C_pulm_art], dim=1).unsqueeze(1)
        decay_res_pulm = (dPrpap_dt + Prpap / (R_pulm_eq * C_pulm_sum + 1e-6)) / (Prpap + 1.0)
        loss_pulm = torch.sum((decay_res_pulm * diastole_mask) ** 2) / (torch.sum(diastole_mask) + 1e-6)

        # [Loss 6] 容积守恒
        P_all = w_loss[:, 0:21, :]
        V_start = torch.sum(C_pred * P_all[:, :, 0], dim=1)
        V_end = torch.sum(C_pred * P_all[:, :, -1], dim=1)
        loss_mass = torch.mean(((V_start - V_end) / (V_start + 1.0)) ** 2)

        # 🌟 最终汇总
        stat_loss = (self.l_data * loss_data) + \
                    (self.l_cos * loss_cos) + \
                    (self.l_fft * loss_fft)

        # 物理项: 由 phys_weight 控制介入
        phys_loss = (self.l_int * loss_integral) + \
                    (self.l_decay * loss_decay) + \
                    (self.l_pulm * loss_pulm) + \
                    (self.l_mass * loss_mass)

        total_loss = stat_loss + phys_weight * phys_loss

        return total_loss, loss_data

# ================= 3. 辅助可视化函数 =================
def print_progress_map(t_idx):
    test_pos = (t_idx + 8) % 10
    val_pos = (t_idx + 7) % 10
    status = [" [X] " if i == test_pos else " [V] " if i == val_pos else "  T  " for i in range(10)]
    print(f"\n" + "═" * 70)
    print(f"📊 进度确认：第 {t_idx:2d} 次循环大训练 (train_{t_idx})\n🗺️ 数据阵型：{''.join(status)}")
    print("═" * 70)


# ================= 4. 主训练函数 =================
def train_loop():
    if not os.path.exists(STRATEGY_PATH):
        print(f"❌ 找不到战略地图: {STRATEGY_PATH}")
        return

    meta = np.load(STRATEGY_PATH, allow_pickle=True).item()
    cv_plan, norm_meta = meta['cv_plan'], meta['norm_meta']

    y_min_t = torch.tensor(norm_meta['y_min'], dtype=torch.float32).to(DEVICE)
    y_max_t = torch.tensor(norm_meta['y_max'], dtype=torch.float32).to(DEVICE)

    test_registry = {}
    registry_path = os.path.join(MOD_DIR, "test_registry.json")
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            test_registry = json.load(f)
        print(f"🔄 加载已有测试记录: {list(test_registry.keys())}")

    # 🌟 实例化稳健版 Loss (权重按照你的模板设定)
    criterion = PhysicsInformedLoss(
        y_min=y_min_t,
        y_max=y_max_t,
        dt=0.005,
        lambdas=(1.0, 1.0, 0.01, 0.001, 0.001, 0.001)
    ).to(DEVICE)

    # 循环控制：如果从头开始用 range(1, 11)
    for t_idx in range(1, 3):
        train_folder = os.path.join(MOD_DIR, f"train_{t_idx}")
        if not os.path.exists(train_folder): os.makedirs(train_folder)
        print_progress_map(t_idx)

        all_source_paths, all_val_paths, all_test_paths = [], [], []
        for group_name in cv_plan.keys():
            fold_data = cv_plan[group_name][t_idx - 1]
            all_source_paths.extend(fold_data['train'])
            all_val_paths.extend(fold_data['val'])
            all_test_paths.extend(fold_data['test'])

        test_registry[f"train_{t_idx}"] = all_test_paths

        source_loader = DataLoader(PhysioDataset(all_source_paths, norm_meta, is_train=True), batch_size=BATCH_SIZE,
                                   shuffle=True)
        val_loader = DataLoader(PhysioDataset(all_val_paths, norm_meta, is_train=False), batch_size=BATCH_SIZE,
                                shuffle=False)

        # 🌟 实例化 22 通道模型
        model = TwinFusionNet(n_channels=22, n_features=2, n_outputs=44).to(DEVICE)

        # --- Stage A: Source 预训练 ---
        print(f"\n▶️ [Stage A] 开始 8000 例 Source 预训练...")
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(1, EPOCHS_PRE + 1):
            current_phys_weight = min(1.0, max(0.0, (epoch - 10) / 20.0))
            model.train()
            total_loss, pure_data_loss = 0.0, 0.0

            pbar = tqdm(source_loader, desc=f"Epoch {epoch:2d}/{EPOCHS_PRE}", leave=False)
            for w, v, y, w_loss in pbar:
                if w.shape[1] > 22: w = w[:, :22, :]
                w, v, y, w_loss = w.to(DEVICE), v.to(DEVICE), y.to(DEVICE), w_loss.to(DEVICE)
                optimizer.zero_grad()
                preds = model(w, v)
                loss, l_data = criterion(preds, y, w_loss, phys_weight=current_phys_weight)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * w.size(0)
                pure_data_loss += l_data.item() * w.size(0)

            print(
                f"   [Train_{t_idx}] Ep {epoch:2d} | Phys_W: {current_phys_weight:.2f} | Loss: {total_loss / len(source_loader.dataset):.5f} | Data: {pure_data_loss / len(source_loader.dataset):.5f}")

        torch.save(model.state_dict(), os.path.join(train_folder, "source_model.pth"))

        # --- Stage B: Val 二次微调 ---
        print(f"\n▶️ [Stage B] 开始 1000 例 Val 二次微调...")
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        for epoch in range(1, EPOCHS_FINE + 1):
            model.train()
            total_loss, pure_data_loss = 0.0, 0.0

            pbar = tqdm(val_loader, desc=f"Epoch {epoch:2d}/{EPOCHS_FINE}", leave=False)
            for w, v, y, w_loss in pbar:
                if w.shape[1] > 22: w = w[:, :22, :]
                w, v, y, w_loss = w.to(DEVICE), v.to(DEVICE), y.to(DEVICE), w_loss.to(DEVICE)
                optimizer.zero_grad()
                preds = model(w, v)
                loss, l_data = criterion(preds, y, w_loss, phys_weight=1.0)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * w.size(0)
                pure_data_loss += l_data.item() * w.size(0)

            print(
                f"   [Train_{t_idx}] Val Ep {epoch:2d} | Loss: {total_loss / len(val_loader.dataset):.5f} | Data: {pure_data_loss / len(val_loader.dataset):.5f}")

        torch.save(model.state_dict(), os.path.join(train_folder, "val_model.pth"))

        with open(registry_path, "w") as f:
            json.dump(test_registry, f)

    print("\n" + "═" * 70 + "\n🏆 10轮大循环物理引擎训练圆满结束！\n" + "═" * 70)


if __name__ == "__main__":
    train_loop()