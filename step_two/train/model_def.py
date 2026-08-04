import torch
import torch.nn as nn
import torch.nn.functional as F

# 改--30---通道数
# 改--103--通道数


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
    def __init__(self, n_channels=11, n_features=2, n_outputs=44):
        super(TwinFusionNet, self).__init__()

        # --- 第一通路：时序卷积 (Waveform Path) ---
        # 【核心升级】扩容卷积层，适配 17 通道的庞大信息流
        self.wave_conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=9, stride=2, padding=4), # 32 -> 64
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),        # 64 -> 128
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),       # 128 -> 256
            nn.BatchNorm1d(256),
            nn.ReLU(),

            # 插入 SE 注意力模块 (通道数同步改为 256)
            SEBlock1d(channels=256, reduction=8),

            nn.AdaptiveAvgPool1d(8),
            nn.Flatten()
        )

        # --- 第二通路：临床指标 (Clinical Path) ---
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # --- 融合公共层 (Shared Representation) ---
        # 【核心同步】256 (波形特征) + 64 (临床特征) = 320
        self.shared_fc = nn.Sequential(
            nn.Linear(2176, 512),  # 之前是 192，现在升级为 320
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )

        # --- 分支 1：电阻 (R) 专属预测头 ---
        self.head_r = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 23)    # 维持 23
        )

        # --- 分支 2：电容 (C) 专属预测头 ---
        self.head_c = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),  # 维持 21
            nn.ReLU(),
            nn.Linear(64, 21)
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
    # 测试一下确保维度闭环
    model = TwinFusionNet(n_channels=11, n_features=2, n_outputs=44)
    w = torch.randn(1, 11, 250)
    v = torch.randn(1, 2)
    output = model(w, v)
    print(f"✅ 维度修正成功！模型输出形态: {output.shape}")
