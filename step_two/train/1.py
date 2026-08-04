import torch
import torch.nn as nn
import torch.nn.functional as F


class TwinFusionNet(nn.Module):
    def __init__(self, n_channels=9, n_features=2, n_outputs=44):  # 1. 这里确保是 2
        super(TwinFusionNet, self).__init__()

        # --- 第一通路：时序卷积 (Waveform Path) ---
        self.wave_conv = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        # --- 第二通路：临床指标 (Clinical Path) ---
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        # --- 融合回归层 ---
        # 128 (CNN) + 64 (MLP) = 192
        self.regressor = nn.Sequential(
            nn.Linear(192, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_outputs),
            nn.Softplus()
        )

    def forward(self, w, v):
        x1 = self.wave_conv(w)
        x2 = self.feat_mlp(v)
        combined = torch.cat((x1, x2), dim=1)
        return self.regressor(combined)


if __name__ == "__main__":
    # --- 2. 这里是报错的根源，必须修改模拟输入的维度 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 实例化模型，传入 n_features=2
    model = TwinFusionNet(n_channels=9, n_features=2, n_outputs=44).to(device)
    print(f"✅ 模型已加载至: {device}")

    # 模拟输入：w 保持不变，但 v 必须改成 (Batch, 2)
    test_w = torch.randn(2, 9, 250).to(device)
    test_v = torch.randn(2, 2).to(device)

    out = model(test_w, test_v)
    print(f"🚀 输出维度: {out.shape}")
    print("✨ 模型结构验证通过！")

