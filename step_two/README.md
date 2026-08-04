step_two/
├── analysis/               # 结果分析与性能评估
│   └── eval_both.py        # 对比评估脚本
├── data/                   # 数据处理模块
│   ├── processed/          # 已处理好的标准化数据集存储目录
│   ├── data_aggregate.py   # 数据聚合：将多分区数据整合为训练格式
│   ├── preprocess_cycles.py# 周期预处理：提取特定心动周期的波形特征
│   └── wave_show.py        # 可视化工具：用于检查压力/流量波形质量
└── train/                  # 模型构建与训练核心
    ├── model_def.py        # 神经网络架构定义（如 CNN+SEBlock 或 LSTM）
    ├── dataset_loader.py   # 自定义数据集加载器
    ├── train_source.py     # 源域/基础模型训练
    ├── train_val.py        # 训练过程中的验证与权重保存
    ├── source_model.pth    # 预训练模型权重
    ├── val_model.pth       # 验证集最优模型权重
    └── 1.py, 2.py, 3.py    # 不同神经网络结构模块