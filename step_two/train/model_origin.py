import torch
import torch.nn as nn
import torch.nn.functional as F


# ================= 1. 轻量级 SE 注意力模块 =================
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


# ================= 2. 改进后的 TwinFusionNet =================
class TwinFusionNet(nn.Module):
    def __init__(self, n_channels=22, n_features=2, n_outputs=44):
        super(TwinFusionNet, self).__init__()

        # --- 第一通路：时序卷积 (提取动态波形细节) ---
        # 减少了一层深度，增加宽度，防止信息在深层过度坍缩
        self.wave_conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # 在中层引入 SE 注意力，过滤 N 通道的冗余信息
            SEBlock1d(channels=128, reduction=8),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        # --- 第二通路：静态特征处理 (心率与 PWTT) ---
        # 增加神经元数量，使静态特征具备更强的“调节”能力
        self.feat_mlp = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Sigmoid()  # 使用 Sigmoid 产生 0-1 的调节门控信号
        )

        # --- 融合与映射层 ---
        # 128 (波形) + 128 (静态门控) = 256
        self.shared_fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # --- 分支 1：电阻 (R) 预测头 ---
        # 额外接入原始静态特征 v，增强物理约束
        self.head_r = nn.Sequential(
            nn.Linear(256 + n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 23),
            nn.Softplus()  # 确保解耦后的物理参数为正
        )

        # --- 分支 2：电容 (C) 预测头 ---
        self.head_c = nn.Sequential(
            nn.Linear(256 + n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 21),
            nn.Softplus()
        )

    def forward(self, w, v):
        """
        w: 时序波形输入 (Batch, 22, 250)
        v: 静态生理指标 (Batch, 2) -> [HR, PWTT]
        """
        # 1. 提取动态特征
        x_wave = self.wave_conv(w)  # (B, 128)

        # 2. 提取静态门控特征
        x_feat = self.feat_mlp(v)  # (B, 128)

        # 3. 条件融合 (Conditional Fusion)
        # 让心率和传导时间决定波形特征的“表达强度”
        # 这一步是解决“垂直条带”的关键，通过 v 的差异强行拉开 x_wave 的分布
        gated_wave = x_wave * x_feat

        combined = torch.cat((gated_wave, x_feat), dim=1)  # (B, 256)
        shared_feat = self.shared_fc(combined)  # (B, 256)

        # 4. 跳跃连接：将原始静态指标 v 再次注入预测头
        # 确保模型在预测 44 个参数时，始终锚定在基础生理基准上
        final_input = torch.cat((shared_feat, v), dim=1)  # (B, 258)

        out_r = self.head_r(final_input)
        out_c = self.head_c(final_input)

        # 5. 拼装返回 (23R + 21C = 44)
        return torch.cat((out_r, out_c), dim=1)


if __name__ == "__main__":
    # 模拟 22 通道输入环境
    n_ch = 22
    n_f = 2
    model = TwinFusionNet(n_channels=n_ch, n_features=n_f, n_outputs=44)

    # 构造测试数据
    test_wave = torch.randn(1, n_ch, 250)
    test_static = torch.randn(1, n_f)

    output = model(test_wave, test_static)

    print("-" * 30)
    print(f"输入通道数: {n_ch}")
    print(f"输出维度: {output.shape} (应为 44)")
    print("-" * 30)