import os
import sys
import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm

# 改80循环几次的结果

# =================================================================
# 核心代码所在的目录
ABS_TRAIN_PATH = r"D:\Digital_twin\step_two\train"
# 11通道模型所在的根目录
ABS_MODEL_ROOT = r"D:\Digital_twin\step_two\mod_1_11_no"
# 策略文件（获取归一化参数 norm_meta）
ABS_STRATEGY_PATH = r"D:\Digital_twin\step_two\data\processed_all\cv_strategy.npy"

# 目标保存目录 (Step 3 的数据中心)
ABS_SAVE_DIR = r"D:\Digital_twin\step_three\all_data"
# 导出的文件名
SAVE_NAME = "RE_01_11.csv"

# =================================================================
# 确保 Python 优先搜索这个绝对路径
if ABS_TRAIN_PATH not in sys.path:
    sys.path.insert(0, ABS_TRAIN_PATH)

from model_def import TwinFusionNet
from dataset_loader import PhysioDataset, DataLoader

# =================================================================
# 3. 核心提取逻辑
# =================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHANNELS = 11  # mod_1_11 专用

# 严格对齐 44 维输出的参数列表 (前23个是R, 后21个是C)
PARAM_NAMES = [
    # R: 索引 0~22
    'R_m', 'R_a', 'R_haa', 'R_lna', 'R_lca', 'R_aop', 'R_rula', 'R_rica', 'R_lica', 'R_lula',
    'R_rsv', 'R_rijv', 'R_lijv', 'R_lsv', 'R_sv', 'R_t', 'R_p', 'R_rpap', 'R_lpap', 'R_rpad',
    'R_lpad', 'R_rpv', 'R_lpv',

    # C: 索引 23~43
    'C_haa', 'C_lna', 'C_lca', 'C_aop', 'C_rula', 'C_rica', 'C_lica', 'C_lula', 'C_sap',
    'C_rsv', 'C_rijv', 'C_lijv', 'C_lsv', 'C_sv', 'C_vc', 'C_rpap', 'C_lpap', 'C_rpad',
    'C_lpad', 'C_rpv', 'C_lpv'
]


def run_extraction():
    # 检查基础文件
    if not os.path.exists(ABS_STRATEGY_PATH):
        print(f"❌ 找不到策略文件: {ABS_STRATEGY_PATH}")
        return
    if not os.path.exists(ABS_SAVE_DIR):
        os.makedirs(ABS_SAVE_DIR)

    # 1. 加载归一化元数据
    meta_data = np.load(ABS_STRATEGY_PATH, allow_pickle=True).item()
    norm_meta = meta_data['norm_meta']

    # 🌟 提取全局极值，准备用于反归一化
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
    print(f"🚀 开始提取数据...")

    # 3. 循环 10 折
    for i in range(1, 11):
        fold_name = f"train_{i}"
        weight_path = os.path.join(ABS_MODEL_ROOT, fold_name, "val_model.pth")

        if not os.path.exists(weight_path):
            print(f"⚠️ 跳过 {fold_name}: 权重文件不存在")
            continue

        # 初始化模型 (确保传入了 n_features=2, n_outputs=44 如果在类定义里需要的话)
        model = TwinFusionNet(n_channels=CHANNELS, n_features=2, n_outputs=44).to(DEVICE)
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE, weights_only=True))
        model.eval()

        # 获取测试集文件
        test_files = registry.get(fold_name, [])
        if not test_files: continue

        test_loader = DataLoader(PhysioDataset(test_files, norm_meta, is_train=False), batch_size=64)

        fold_apes = []
        with torch.no_grad():
            for w, v, y, _ in test_loader:
                # 截断保护
                if w.shape[1] > CHANNELS: w = w[:, :CHANNELS, :]

                # 预测输出和真实标签均为 [0, 1] 之间的值
                preds_norm = model(w.to(DEVICE), v.to(DEVICE)).cpu().numpy()
                trues_norm = y.numpy()

                # 🌟 核心：两步反归一化 (先线性还原对数，再取指数还原物理值)
                # 第一步：还原到 log(y) 空间
                preds_log = preds_norm * (y_log_max - y_log_min + 1e-8) + y_log_min
                trues_log = trues_norm * (y_log_max - y_log_min + 1e-8) + y_log_min

                # 第二步：还原到物理空间 (Exp)
                preds_phys = np.exp(preds_log)
                trues_phys = np.exp(trues_log)

                # 🌟 在真实的物理空间计算相对百分比误差 APE (%)
                # 乘以 100 方便后续直接用作百分比可视化，分母加极小数防止除 0
                ape = np.abs((preds_phys - trues_phys) / (trues_phys + 1e-10)) * 100
                fold_apes.append(ape)

        if fold_apes:
            df_fold = pd.DataFrame(np.vstack(fold_apes), columns=PARAM_NAMES)
            df_fold['Model_Type'] = "Mod_11"
            df_fold['Fold'] = i
            all_fold_records.append(df_fold)
            print(f"   ✅ {fold_name} 提取并反归一化成功")

    # 4. 合并并保存到 Step 3
    if all_fold_records:
        final_df = pd.concat(all_fold_records, ignore_index=True)
        output_full_path = os.path.join(ABS_SAVE_DIR, SAVE_NAME)
        final_df.to_csv(output_full_path, index=False)
        print(f"\n✨ 提取大功告成！全量数据已存入: {output_full_path}")
    else:
        print("❌ 未能提取任何数据，请检查子文件夹是否包含 val_model.pth")


if __name__ == "__main__":
    run_extraction()