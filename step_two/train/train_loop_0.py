import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from tqdm import tqdm
from model_def import TwinFusionNet
from dataset_loader import PhysioDataset, DataLoader

# ================= 1. 基础配置 =================
ROOT_DIR = r"D:\Digital_twin\step_two"
MOD_DIR = os.path.join(ROOT_DIR, "mod_1_11")
STRATEGY_PATH = os.path.join(ROOT_DIR, r"data\processed_all\cv_strategy.npy")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS_PRE = 50  # Step A: Source 预训练轮数
EPOCHS_FINE = 20  # Step B: Val 二次微调轮数


# ================= 2. 辅助可视化函数 =================
def print_progress_map(t_idx):
    """
    可视化阵型图打印：T=Source(预训练), V=Val(微调), X=Test(绝密验证)
    """
    # 映射逻辑：每一轮滚动一个位置
    # 逻辑：Test 占 1 份，Val 占 1 份，Train 占 8 份
    test_pos = (t_idx + 8) % 10
    val_pos = (t_idx + 7) % 10

    status = []
    for i in range(10):
        if i == test_pos:
            status.append(" [X] ")  # 隔离考试
        elif i == val_pos:
            status.append(" [V] ")  # 二次微调
        else:
            status.append("  T  ")  # 基础预训练

    bar = "".join(status)
    print(f"\n" + "═" * 70)
    print(f"📊 进度确认：第 {t_idx:2d} 次循环大训练 (train_{t_idx})")
    print(f"🗺️ 数据阵型：{bar}")
    print(f"📈 规模：Source(8000例) | Val(1000例) | Test(1000例隔离)")
    print("═" * 70)


# ================= 3. 主训练函数 =================
def train_loop():
    # A. 资源检查
    if not os.path.exists(STRATEGY_PATH):
        print("❌ 错误：找不到 cv_strategy.npy，请确保已运行规划脚本。")
        return

    meta = np.load(STRATEGY_PATH, allow_pickle=True).item()
    cv_plan = meta['cv_plan']
    norm_meta = meta['norm_meta']
    test_registry = {}  # 记录考卷

    # B. 开启 10 轮滚动大循环
    for t_idx in range(1, 11):
        train_folder = os.path.join(MOD_DIR, f"train_{t_idx}")
        if not os.path.exists(train_folder): os.makedirs(train_folder)

        # 1. 打印阵型图
        print_progress_map(t_idx)

        # 2. 全分区路径集结 (8组汇聚)
        all_source_paths, all_val_paths, all_test_paths = [], [], []
        for group_name in cv_plan.keys():
            fold_data = cv_plan[group_name][t_idx - 1]
            all_source_paths.extend(fold_data['train'])  # 8000
            all_val_paths.extend(fold_data['val'])  # 1000
            all_test_paths.extend(fold_data['test'])  # 1000

        # 记录本轮对应的隔离考卷
        test_registry[f"train_{t_idx}"] = all_test_paths

        # 3. 构造 DataLoader
        source_loader = DataLoader(PhysioDataset(all_source_paths, norm_meta, is_train=True), batch_size=BATCH_SIZE,
                                   shuffle=True)
        val_loader = DataLoader(PhysioDataset(all_val_paths, norm_meta, is_train=True), batch_size=BATCH_SIZE,
                                shuffle=True)

        # 4. 初始化模型 (R=23, C=21)
        model = TwinFusionNet(n_channels=11, n_features=2, n_outputs=44).to(DEVICE)
        criterion = nn.MSELoss()

        # ---------------------------------------------------------
        # 阶段 A: Source 预训练 (8000例)
        # ---------------------------------------------------------
        print(f"\n▶️ [Stage A] 开始 8000 例 Source 预训练...")
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(1, EPOCHS_PRE + 1):
            model.train()
            total_loss = 0.0
            # 使用 tqdm 增加 Epoch 内的进度条
            pbar = tqdm(source_loader, desc=f"Epoch {epoch:2d}/{EPOCHS_PRE}", leave=False)
            for w, v, y in pbar:
                w, v, y = w.to(DEVICE), v.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(w, v), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * w.size(0)

            avg_loss = total_loss / len(source_loader.dataset)
            # 打印第 N 次训练的 Loss
            print(f"   [Train_{t_idx}] Source Phase | Epoch {epoch:2d} | LOSS = {avg_loss:.8f}")

        torch.save(model.state_dict(), os.path.join(train_folder, "source_model.pth"))

        # ---------------------------------------------------------
        # 阶段 B: Val 二次微调 (1000例)
        # ---------------------------------------------------------
        print(f"\n▶️ [Stage B] 开始 1000 例 Val 二次微调...")
        optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 降低学习率

        for epoch in range(1, EPOCHS_FINE + 1):
            model.train()
            total_loss = 0.0
            pbar = tqdm(val_loader, desc=f"Epoch {epoch:2d}/{EPOCHS_FINE}", leave=False)
            for w, v, y in pbar:
                w, v, y = w.to(DEVICE), v.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(w, v), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * w.size(0)

            avg_loss = total_loss / len(val_loader.dataset)
            print(f"   [Train_{t_idx}] Val Phase    | Epoch {epoch:2d} | LOSS = {avg_loss:.8f}")

        torch.save(model.state_dict(), os.path.join(train_folder, "val_model.pth"))

    # 5. 生成阅卷手册
    with open(os.path.join(MOD_DIR, "test_registry.json"), "w") as f:
        json.dump(test_registry, f)

    print("\n" + "═" * 70)
    print("🏆 全部 10 轮大循环训练圆满结束！")
    print(f"📂 模型保存目录：{MOD_DIR}")
    print(f"📑 测试注册表：{os.path.join(MOD_DIR, 'test_registry.json')}")
    print("═" * 70)


if __name__ == "__main__":
    train_loop()