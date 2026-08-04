Digital_twin/
├── step_one/                  # 得到心肺物理模型扰乱后的数据
│   ├── dataset_10000/               # 数据结果
│   ├── BatchRun_Save_8group.m       # 心肺模型模块扰动
│   ├── Entry_for_NormalHumanCirculationSystem        # 正常人心肺运行
├── step_two/                   # 输入为 11 通道数据
│   ├── data/                        # step_one 中得到的数据预处理
│   ├── train/                       # 模型构建与训练核心
│   ├── analysis/                    # 结果分析与性能评估
│   ├── model_1_11_no/               # 训练结果
├──  step_two_2/                # 输入为 22 通道数据
│   ├── data/                        # step_one 中得到的数据预处理
│   ├── train/                       # 模型构建与训练核心
│   ├── analysis/                    # 结果分析与性能评估
│   ├── model_2_22_no/               # 训练结果
├── step_three/                 # 训练结果
│   ├── all_data_no/                 # 提取需要的对应数据
│   ├── compare/                     # 对比11通道与22通道训练结果
│   └──  scattered_points/           # 11通道与22通道各自训练结果
└── plot/                       # 可视化结果
...


# 以上为部分主要代码文件注释~
# step_one中的代码主要在 MATLAB中运行的，关于心肺系统物理原模型在另一个压缩包里