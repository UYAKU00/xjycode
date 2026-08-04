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
import matplotlib.pyplot as plt



def plot_training_history(history, save_path):
    """可视化工具：在不影响训练的情况下生成曲线 [cite: 250]"""
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 6))
    # 使用对数坐标，因为 PINN 的 Loss 通常在 1e-4 到 1e-8 之间 [cite: 235, 244]
    plt.semilogy(epochs, history["train_loss"], label='Train Loss (SmoothL1)', color='#1f77b4')
    plt.semilogy(epochs, history["val_loss"], label='Val Loss (Monitor)', color='#ff7f0e')

    if len(epochs) > 50:
        plt.axvline(x=50, color='gray', linestyle='--', label='Stage B Start')

    plt.xlabel('Total Epochs')
    plt.ylabel('Loss Value (Log Scale)')
    plt.title('Physics-Informed Training Convergence')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(save_path)
    plt.close()



def run_evaluation(model, loader, criterion, device):
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for w, v, y, w_loss in loader:
            w, v, y, w_loss = w.to(device), v.to(device), y.to(device), w_loss.to(device)
            preds = model(w, v)
            # 使用固定物理权重客观评估 [cite: 112, 130]
            loss, _ = criterion(preds, y, w_loss, phys_weight=0.5)
            total_val_loss += loss.item() * w.size(0)
    return total_val_loss / len(loader.dataset)


# -----240-循环次数--------
# -----257改通道数----------

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
    def __init__(self, y_log_min, y_log_max, dt=0.005, lambdas=(1.0, 0, 0.005, 0.001, 0.002, 0.001)):
        super(PhysicsInformedLoss, self).__init__()
        self.dt = dt
        self.y_log_min = y_log_min
        self.y_log_max = y_log_max

        self.l_data, self.l_cos, self.l_fft, self.l_int, self.l_pulm, self.l_mass = lambdas
        self.l_decay = 0.001

        #---------参数权重----------
        good_idx = [0, 2, 3, 4, 5, 6, 7, 10, 11, 14, 15, 23, 24, 28]
        mid_idx = [9, 13, 21, 22, 25, 26, 29, 32, 33, 40, 41, 42, 43]
        bad_idx = [1, 8, 12, 16, 17, 18, 19, 20, 27, 30, 31, 34, 35, 36, 37, 38, 39]

        self.register_buffer("good_idx", torch.tensor(good_idx, dtype=torch.long))
        self.register_buffer("mid_idx", torch.tensor(mid_idx, dtype=torch.long))
        self.register_buffer("bad_idx", torch.tensor(bad_idx, dtype=torch.long))

        # 🌟 新增：组合参数监督索引 (已根据剔除 Rsap 和 Rvc 后的 44 维向量对齐)
        # A: 肺循环总阻力 (Rp, Rrpap, Rlpap, Rrpad, Rlpad)
        self.register_buffer("idx_R_pulm_comb", torch.tensor([16, 17, 18, 19, 20], dtype=torch.long))
        # B: 静脉总顺应性 (Crsv, Crijv, Clijv, Csv, Cvc)
        self.register_buffer("idx_C_venous_comb", torch.tensor([32, 33, 34, 35, 36], dtype=torch.long))
        # C: 远端系统阻力 (Ra, Rlica, Rlijv)
        self.register_buffer("idx_R_dist_comb", torch.tensor([1, 8, 12], dtype=torch.long))
        # D: 肺部总顺应性 (Crpap, Clpap, Crpad, Clpad)
        self.register_buffer("idx_C_pulm_comb", torch.tensor([37, 38, 39, 40], dtype=torch.long))


    def forward(self, preds, targets, w_loss, phys_weight=1.0):

        # --- 1. 基础数据损失与样本加权 (解决垂直条带的关键) ---
        # loss_data = 0.7 * F.mse_loss(preds, targets) + 0.3 * F.smooth_l1_loss(preds, targets)
        mse_per_param = (preds - targets) ** 2
        smooth_per_param = F.smooth_l1_loss(preds, targets, reduction='none')

        base_loss = 0.7 * mse_per_param + 0.3 * smooth_per_param

        loss_good = base_loss[:, self.good_idx].mean()
        loss_mid = base_loss[:, self.mid_idx].mean()
        loss_bad = base_loss[:, self.bad_idx].mean()

        loss_data = 1.0 * loss_good + 0.8 * loss_mid + 0.5 * loss_bad

        '''
        loss_data = 1.0 * loss_good + 0.8 * loss_mid + 0.3 * loss_bad
        '''

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
        preds_log = preds * (self.y_log_max - self.y_log_min + 1e-8) + self.y_log_min
        preds_phys = torch.exp(preds_log)



        # 🌟 新增：真实值还原 (用于组合监督对比)
        targets_log = targets * (self.y_log_max - self.y_log_min + 1e-8) + self.y_log_min
        targets_phys = torch.exp(targets_log)

        # 🌟 核心：计算 4 组等效组合监督损失 (使用相对误差 MAPE)
        def calc_comb_loss(p_phys, t_phys, idxs):
            p_sum = torch.sum(p_phys[:, idxs], dim=1)
            t_sum = torch.sum(t_phys[:, idxs], dim=1)
            return torch.mean(((p_sum - t_sum) / (t_sum + 1e-6)) ** 2)

        loss_comb = calc_comb_loss(preds_phys, targets_phys, self.idx_R_pulm_comb) + \
                    calc_comb_loss(preds_phys, targets_phys, self.idx_C_venous_comb) + \
                    calc_comb_loss(preds_phys, targets_phys, self.idx_R_dist_comb) + \
                    calc_comb_loss(preds_phys, targets_phys, self.idx_C_pulm_comb)

        # 🌟 如果预测的组合物理值超出了训练集的最大/最小范围，给予额外重罚
        R_pulm_sum_pred = torch.sum(preds_phys[:, self.idx_R_pulm_comb], dim=1)
        R_pulm_limit_high = self.y_log_max[self.idx_R_pulm_comb].exp().sum()
        R_pulm_limit_low = self.y_log_min[self.idx_R_pulm_comb].exp().sum()
        # 这里的 R_pulm_sum_pred 也要确保索引名一致
        R_pulm_sum_pred = torch.sum(preds_phys[:, self.idx_R_pulm_comb], dim=1)
        # 这种惩罚能防止模型为了凑 Loss 把某个参数预测成天文数字，而另一个预测成负数
        loss_range = torch.mean(F.relu(R_pulm_sum_pred - R_pulm_limit_high)) + \
                     torch.mean(F.relu(R_pulm_limit_low - R_pulm_sum_pred))
        loss_comb = loss_comb + 0.3 * loss_range


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

        # 最终汇总
        stat_loss = (self.l_data * loss_data) + \
                    (self.l_cos * loss_cos) + \
                    (self.l_fft * loss_fft)

        # 物理项: 由 phys_weight 控制介入
        phys_loss = (self.l_int * loss_integral) + \
                    (self.l_decay * loss_decay) + \
                    (self.l_pulm * loss_pulm) + \
                    (self.l_mass * loss_mass)

        total_loss = stat_loss + (0.05 * loss_comb) + phys_weight * phys_loss

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
        print("❌ 找不到战略地图！")
        return

    meta = np.load(STRATEGY_PATH, allow_pickle=True).item()
    cv_plan, norm_meta = meta['cv_plan'], meta['norm_meta']

    # 4：提取极值并转换为张量
    y_log_min_t = torch.tensor(norm_meta['y_log_min'], dtype=torch.float32).to(DEVICE)
    y_log_max_t = torch.tensor(norm_meta['y_log_max'], dtype=torch.float32).to(DEVICE)

    # 将极值传递给物理损失函数
    criterion = PhysicsInformedLoss(
        y_log_min=y_log_min_t,
        y_log_max=y_log_max_t,
        dt=0.005,
        lambdas=(1.0, 0, 0.005, 0.001, 0.002, 0.001)
    ).to(DEVICE)

    # 🌟 重点修正位置：确保 test_registry 在循环外初始化
    test_registry = {}
    registry_path = os.path.join(MOD_DIR, "test_registry.json")

    if os.path.exists(registry_path):
        try:
            with open(registry_path, "r") as f:
                test_registry = json.load(f)
            print(f"🔄 成功加载已有测试记录，目前包含: {list(test_registry.keys())}")
        except Exception as e:
            print(f"⚠️ 加载旧记录失败，将创建新记录: {e}")
            test_registry = {}  # 如果读取失败则重置为空
    else:
        print("🆕 未发现旧记录，将创建全新的 test_registry")


    for t_idx in range(1, 11):
        train_folder = os.path.join(MOD_DIR, f"train_{t_idx}")
        if not os.path.exists(train_folder): os.makedirs(train_folder)
        print_progress_map(t_idx)

        # 🌟 核心修改：动态集结所有分区的路径，不再假设 Group 名称
        all_pre_paths, all_fine_paths, all_val_paths, all_test_paths = [], [], [], []

        for g_name in cv_plan.keys():
            plan = cv_plan[g_name][t_idx - 1]

            # 这里的划分比例必须与 dataset.py 中的逻辑严格对齐 (8:2)
            full_train = plan['train']
            split_point = int(len(full_train) * 0.8)

            all_pre_paths.extend(full_train[:split_point])
            all_fine_paths.extend(full_train[split_point:])
            all_val_paths.extend(plan['val'])
            all_test_paths.extend(plan['test'])

        # 直接在 train_loop 中构建全量加载器
        pre_loader = DataLoader(PhysioDataset(all_pre_paths, norm_meta, is_train=True),
                                batch_size=BATCH_SIZE, shuffle=True)
        fine_loader = DataLoader(PhysioDataset(all_fine_paths, norm_meta, is_train=True),
                                 batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(PhysioDataset(all_val_paths, norm_meta, is_train=False),
                                batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(PhysioDataset(all_test_paths, norm_meta, is_train=False),
                                 batch_size=BATCH_SIZE, shuffle=False)

        # 初始化模型 (22通道输入)
        model = TwinFusionNet(n_channels=22, n_features=2, n_outputs=44).to(DEVICE)
        best_val_loss = float('inf')

        # 初始化字典
        history = {
            "train_loss": [],
            "val_loss": [],
            "phys_weight": []
        }

        # ---------------------------------------------------------
        # 阶段 A: Source 预训练
        # ---------------------------------------------------------
        print(f"\n▶️ [Stage A] 正在训练 80% 混合分区子集 (样本数: {len(all_pre_paths)})...")
        optimizer = optim.Adam(model.parameters(), lr=0.0003)

        for epoch in range(1, EPOCHS_PRE + 1):
            current_phys_weight = min(0.5, max(0.0, (epoch - 30) / 20.0))
            model.train()

            total_loss = 0.0
            pure_data_loss = 0.0

            pbar = tqdm(pre_loader, desc=f"Pre-train {epoch}", leave=False)
            for w, v, y, w_loss in pbar:
                w, v, y, w_loss = w.to(DEVICE), v.to(DEVICE), y.to(DEVICE), w_loss.to(DEVICE)
                optimizer.zero_grad()
                preds = model(w, v)
                loss, l_data = criterion(preds, y, w_loss, phys_weight=current_phys_weight)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * w.size(0)
                pure_data_loss += l_data.item() * w.size(0)

            # 🌟 记录预训练阶段数据
            avg_data_loss = pure_data_loss / len(pre_loader.dataset)
            val_score = run_evaluation(model, val_loader, criterion, DEVICE)
            history["train_loss"].append(avg_data_loss)  # 记录纯数据损失或总损失
            history["val_loss"].append(val_score)
            history["phys_weight"].append(current_phys_weight)

            # 每一轮预训练后跑验证集（裁判打分）
            val_score = run_evaluation(model, val_loader, criterion, DEVICE)

            if val_score < best_val_loss:
                best_val_loss = val_score
                torch.save(model.state_dict(), os.path.join(train_folder, "best_source_model.pth"))

            print(f"   Epoch {epoch:2d} | Val Loss: {val_score:.6f}")


        # ---------------------------------------------------------
        # 阶段 B: Val 二次微调
        # ---------------------------------------------------------
        print(f"\n▶️ [Stage B] 加载最佳预训练模型，开始 20% 子集微调 (样本数: {len(all_fine_paths)})...")
        model.load_state_dict(torch.load(os.path.join(train_folder, "best_source_model.pth")))
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        for epoch in range(1, EPOCHS_FINE + 1):
            model.train()

            total_loss = 0.0
            pure_data_loss = 0.0

            pbar = tqdm(fine_loader, desc=f"Fine-tune {epoch}", leave=False)
            for w, v, y, w_loss in pbar:
                w, v, y, w_loss = w.to(DEVICE), v.to(DEVICE), y.to(DEVICE), w_loss.to(DEVICE)
                optimizer.zero_grad()
                preds = model(w, v)

                loss, l_data = criterion(preds, y, w_loss, phys_weight=0.5)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * w.size(0)
                pure_data_loss += l_data.item() * w.size(0)

            # 🌟 插入位置：记录微调阶段数据
            avg_data_loss = pure_data_loss / len(fine_loader.dataset)
            val_score = run_evaluation(model, val_loader, criterion, DEVICE)
            history["train_loss"].append(avg_data_loss)
            history["val_loss"].append(val_score)
            history["phys_weight"].append(0.5)

            val_score = run_evaluation(model, val_loader, criterion, DEVICE)

            if val_score < best_val_loss:
                best_val_loss = val_score
                torch.save(model.state_dict(), os.path.join(train_folder, "val_model.pth"))

            print(f"   Val Epoch {epoch:2d} | Val Loss: {val_score:.6f}")

        # 记录该折的测试路径，方便后续独立测试
        test_registry[f"train_{t_idx}"] = all_test_paths


        # 🌟 插入位置：在记录测试路径之前，把这一折的“黑匣子”存了
        with open(os.path.join(train_folder, "history.json"), "w") as f:
            json.dump(history, f)
        # 自动生成 Loss 曲线图
        plot_training_history(history, os.path.join(train_folder, "loss_curve.png"))
        # 原有代码：记录测试路径
        test_registry[f"train_{t_idx}"] = all_test_paths


        with open(registry_path, "w") as f:
            json.dump(test_registry, f)


    print("\n" + "═" * 70)
    print("🏆 全部 10 轮大循环物理引擎训练圆满结束！")
    print("═" * 70)


if __name__ == "__main__":
    # 1. 确保日志目录存在
    if not os.path.exists(MOD_DIR):
        os.makedirs(MOD_DIR)

    # 2. 打印当前环境信息，确认是 22 通道配置 [cite: 10, 13]
    print(f"🚀 系统启动 | 设备: {DEVICE} | 模式: 8:2:1:1 架构")
    print(f"📂 模型保存路径: {MOD_DIR}")

    # 3. 运行主训练循环
    try:
        train_loop()
        print("\n✨ [Done] 10折交叉验证及两阶段物理反演任务已全部完成。")
    except Exception as e:
        print(f"💥 运行崩溃: {e}")