import torch
import torch.nn as nn
import torch.nn.functional as F


# ================= 1. 定义轻量级 SE 注意力模块 =================
class SEBlock1d(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


# ================= 2. 双分支多任务网络 =================
class TwinFusionNet(nn.Module):
    # 恢复了 n_outputs=44，完美兼容你的训练脚本！
    def __init__(self, n_channels=9, n_features=2, n_outputs=44):
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

            # 插入 SE 注意力模块
            SEBlock1d(channels=128, reduction=8),

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

        # --- 融合公共层 (Shared Representation) ---
        self.shared_fc = nn.Sequential(
            nn.Linear(192, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # --- 分支 1：电阻 (R) 专属预测头 ---
        self.head_r = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 23),  # 内部写死输出 23 维
            nn.Softplus()
        )

        # --- 分支 2：电容 (C) 专属预测头 ---
        self.head_c = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 21),  # 内部写死输出 21 维
            nn.Softplus()
        )

    def forward(self, w, v):
        x1 = self.wave_conv(w)
        x2 = self.feat_mlp(v)

        combined = torch.cat((x1, x2), dim=1)
        shared_feat = self.shared_fc(combined)

        out_r = self.head_r(shared_feat)
        out_c = self.head_c(shared_feat)

        # 内部拼装回 44 维返回给外界
        return torch.cat((out_r, out_c), dim=1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟外部调用方式（完全按照你训练脚本里的写法）
    model = TwinFusionNet(n_channels=9, n_features=2, n_outputs=44).to(device)
    print(f"✅ 模型已加载至: {device}")

    test_w = torch.randn(2, 9, 250).to(device)
    test_v = torch.randn(2, 2).to(device)

    out = model(test_w, test_v)
    print(f"🚀 输出维度: {out.shape}")
