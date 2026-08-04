import os
import sys
import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm

# 31改通道数
# 70改循环几次


# =================================================================
# 1. 路径与配置 (锁定 Step_two_2 和 22通道)
# =================================================================
ABS_TRAIN_PATH = r"D:\Digital_twin\step_two_2\train"
ABS_MODEL_ROOT = r"D:\Digital_twin\step_two_2\mod_2_22_no"
ABS_STRATEGY_PATH = r"D:\Digital_twin\step_two_2\data\processed_all\cv_strategy.npy"

# 目标保存目录 (Step 3 的数据中心)
ABS_SAVE_DIR = r"D:\Digital_twin\step_three\all_data"
SAVE_NAME = "RE_02_22.csv"

if ABS_TRAIN_PATH not in sys.path:
    sys.path.insert(0, ABS_TRAIN_PATH)

from model import TwinFusionNet
from dataset import PhysioDataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHANNELS = 22

PARAM_NAMES = [
    'R_m', 'R_a', 'R_haa', 'R_lna', 'R_lca', 'R_aop', 'R_rula', 'R_rica', 'R_lica', 'R_lula',
    'R_rsv', 'R_rijv', 'R_lijv', 'R_lsv', 'R_sv', 'R_t', 'R_p', 'R_rpap', 'R_lpap', 'R_rpad',
    'R_lpad', 'R_rpv', 'R_lpv',
    'C_haa', 'C_lna', 'C_lca', 'C_aop', 'C_rula', 'C_rica', 'C_lica', 'C_lula', 'C_sap',
    'C_rsv', 'C_rijv', 'C_lijv', 'C_lsv', 'C_sv', 'C_vc', 'C_rpap', 'C_lpap', 'C_rpad',
    'C_lpad', 'C_rpv', 'C_lpv'
]


def run_extraction():
    if not os.path.exists(ABS_STRATEGY_PATH):
        print(f"❌ 找不到策略文件: {ABS_STRATEGY_PATH}")
        return
    if not os.path.exists(ABS_SAVE_DIR):
        os.makedirs(ABS_SAVE_DIR)

    # 1. 加载归一化元数据
    meta_data = np.load(ABS_STRATEGY_PATH, allow_pickle=True).item()
    norm_meta = meta_data['norm_meta']

    # 🌟 关键修复 1：根据你的 dataset.py，使用 log 域的极值
    y_log_min = norm_meta['y_log_min']
    y_log_max = norm_meta['y_log_max']

    # 2. 加载测试注册表
    reg_path = os.path.join(ABS_MODEL_ROOT, "test_registry.json")
    if not os.path.exists(reg_path):
        print(f"❌ 找不到注册表: {reg_path}")
        return
    with open(reg_path, 'r') as f:
        registry = json.load(f)

    all_fold_records = []
    print(f"🚀 开始从 22通道模型提取数据...")

    # 3. 循环折数 (根据需要调整 range)
    for i in range(1, 11):  # 示例只跑第1折，需要全量改 range(1, 11)
        fold_name = f"train_{i}"
        weight_path = os.path.join(ABS_MODEL_ROOT, fold_name, "val_model.pth")

        if not os.path.exists(weight_path):
            print(f"⚠️ 跳过 {fold_name}: 权重文件不存在")
            continue

        model = TwinFusionNet(n_channels=CHANNELS, n_features=2, n_outputs=44).to(DEVICE)
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE, weights_only=True))
        model.eval()

        test_files = registry.get(fold_name, [])
        if not test_files: continue

        test_loader = DataLoader(PhysioDataset(test_files, norm_meta, is_train=False), batch_size=64)

        fold_apes = []
        with torch.no_grad():
            for w, v, y, _ in tqdm(test_loader, desc=f"Processing {fold_name}", leave=False):
                w, v, y = w.to(DEVICE), v.to(DEVICE), y.to(DEVICE)

                # 预测输出 (Log 归一化空间)
                preds_norm = model(w, v).cpu().numpy()
                trues_norm = y.cpu().numpy()

                # 🌟 关键修复 2：执行 Log 反归一化
                # 第一步：线性还原到 Log 空间
                preds_log = preds_norm * (y_log_max - y_log_min + 1e-8) + y_log_min
                trues_log = trues_norm * (y_log_max - y_log_min + 1e-8) + y_log_min

                # 第二步：指数还原到原始物理量纲 (Exp)
                preds_phys = np.exp(preds_log)
                trues_phys = np.exp(trues_log)

                # 🌟 计算 APE (%)：(abs(pred - true) / true) * 100
                # 使用真实物理值计算，确保小提琴图反映的是真实误差分布
                ape = np.abs((preds_phys - trues_phys) / (trues_phys + 1e-10)) * 100
                fold_apes.append(ape)

        if fold_apes:
            df_fold = pd.DataFrame(np.vstack(fold_apes), columns=PARAM_NAMES)
            df_fold['Model_Type'] = "Mod_22"
            df_fold['Fold'] = i
            all_fold_records.append(df_fold)
            print(f"   ✅ {fold_name} 提取成功")

    if all_fold_records:
        final_df = pd.concat(all_fold_records, ignore_index=True)
        output_full_path = os.path.join(ABS_SAVE_DIR, SAVE_NAME)
        final_df.to_csv(output_full_path, index=False)
        print(f"\n✨ 22通道原始误差分布已存入: {output_full_path}")
    else:
        print("❌ 未提取到任何数据。")


if __name__ == "__main__":
    run_extraction()