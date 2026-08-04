%% 随机取样仿真
% R:0.85-1.20
% C:0.90-1.15

% 更改后
% R:0.95-1.35
% C:0.75-1.05


% L:0.90-1.15(仅当KL>0)
% 神经调制:FhrS/FhrVE[0.3,0.7](心率筛选50-110bpm)
% Fcon/FvasoE[0.4,0.6]

function BatchRun_PhysiologicalGroups(base_dir)
    if nargin < 1
        base_dir = fullfile('step_one', 'dataset_9000'); 
    end
    if ~exist(base_dir, 'dir'), mkdir(base_dir); end

    %% 1. 定义 9 大生理环节的分组信息 [cite: 103, 104]
    % 结构体包含：对应的 R 索引, 对应的 C 索引, 分区命名
    
    % 组1: 左心与主体循环根部
    groups(1).R_idx = [1, 2, 6];       % Rm, Ra, Raop
    groups(1).C_idx = [4, 9];          % Caop, Csap
    groups(1).Name  = 'L-Heart_Aorta';

    % 组2: 右侧头颈循环
    groups(2).R_idx = [3, 8, 13];      % Rhaa, Rrica, Rrijv
    groups(2).C_idx = [1, 6, 11];      % Chaa, Crica, Crijv
    groups(2).Name  = 'R-Head_Neck';

    % 组3: 右侧上肢循环
    groups(3).R_idx = [7, 12];         % Rrula, Rrsv
    groups(3).C_idx = [5, 10];         % Crula, Crsv
    groups(3).Name  = 'R-Upper_Limb';

    % 组4: 左侧头颈循环
    groups(4).R_idx = [4, 9, 14];      % Rlna, Rlica, Rlijv
    groups(4).C_idx = [2, 7, 12];      % Clna, Clica, Clijv
    groups(4).Name  = 'L-Head_Neck';

    % 组5: 左侧上肢循环
    groups(5).R_idx = [5, 10, 15];     % Rlca, Rlula, Rlsv
    groups(5).C_idx = [3, 8, 13];      % Clca, Clula, Clsv
    groups(5).Name  = 'L-Upper_Limb';

    % 组6: 体静脉回流与右心射血
    groups(6).R_idx = [16, 18, 19];    % Rsv, Rt, Rp
    groups(6).C_idx = [14, 15];        % Csv, Cvc
    groups(6).Name  = 'Venous_R-Heart';

    % 组7: 右侧肺循环
    groups(7).R_idx = [20, 22, 24];    % Rrpap, Rrpad, Rrpv
    groups(7).C_idx = [16, 18, 20];    % Crpap, Crpad, Crpv
    groups(7).Name  = 'R-Lung';

    % 组8: 左侧肺循环
    groups(8).R_idx = [21, 23, 25];    % Rlpap, Rlpad, Rlpv
    groups(8).C_idx = [17, 19, 21];    % Clpap, Clpad, Clpv
    groups(8).Name  = 'L-Lung';

    % 组9: 全身系统性亚健康扰动 (Systemic Perturbation)
    groups(9).R_idx = 1:25;            % 全身 25 个电阻同步扰动
    groups(9).C_idx = 1:21;            % 全身 21 个电容同步扰动
    groups(9).Name  = 'Systemic_Sub-health';

    num_groups = length(groups);
    sims_per_group = 1000; % 每组固定生成 1000 份样本

    %% 2. 自动化仿真配置 [cite: 63, 64]
    start_g = 1;   % 断点续传起始组
    start_k = 1;   % 断点续传起始次数

    fprintf('\n🚀 启动 9,000 组大规模生理仿真任务...\n');

    for g = 1:num_groups
        if g < start_g, continue; end
        
        r_target = groups(g).R_idx;
        c_target = groups(g).C_idx;
        
        % 创建独立文件夹，层次化存储 [cite: 110]
        folder_name = sprintf('Group%d_%s', g, groups(g).Name);
        group_dir = fullfile(base_dir, folder_name);
        if ~exist(group_dir, 'dir'), mkdir(group_dir); end
        
        fprintf('\n>>> 正在处理第 %d/9 组: [%s]\n', g, groups(g).Name);
        
        for k = 1:sims_per_group
            if g == start_g && k < start_k, continue; end
            
            try
                % 设定随机种子以保证实验可重复性
                rng(20260000 + g*1000 + k, 'twister'); 
                
                % 1. 初始化倍率 (1 代表生理基准值)
                R_mult = ones(1, 25);
                C_mult = ones(1, 21);
                
                % 2. 核心：基于对数均匀分布的随机扰动 
                % 修正公式：exp(log(min) + rand*(log(max) - log(min)))
                for r_idx = r_target
                    R_mult(r_idx) = exp(log(0.95) + rand*(log(1.35) - log(0.95)));
                end
                for c_idx = c_target
                    C_mult(c_idx) = exp(log(0.75) + rand*(log(1.05) - log(0.75)));
                end
                
                % 3. 模拟自主神经调节 (Fcon, Fvaso, Hr) [cite: 108]
                Fcon_scalar = 0.4 + 0.2*rand;
                Fvaso_scalar = 0.4 + 0.2*rand;
                
                % 循环筛选，确保心率在健康人静息区间 [50, 110] bpm
                ok = false; tries = 0;
                while ~ok && tries < 100
                    FhrS_scalar = 0.3 + 0.4*rand;
                    FhrV_scalar = 0.3 + 0.4*rand;
                    % 基于非线性多项式的心率映射逻辑
                    Hr = 35 + 140*FhrS_scalar - 40*FhrS_scalar.^2 - 32*FhrV_scalar + 10*FhrV_scalar.^2 - 20*FhrV_scalar.*FhrS_scalar;
                    ok = (Hr >= 50) && (Hr <= 110);
                    tries = tries + 1;
                end
                
                % 4. 调用 0D 集总参数物理模型进行数值求解 [cite: 74, 87]
                % 仿真总长 700s，取稳态平衡后的两个心动周期 
                run('Entry_for_NormalHumanCirculationSystem.m');
                
                % 5. 序列化命名并重采样为 200 点 
                sid = sprintf('Group%d_%04d', g, k);
                SaveTwoCycleSample(sid, group_dir, 200); 
                
                if mod(k, 100) == 0
                    fprintf('   -- 已完成该组第 %04d/1000 次仿真 --\n', k);
                end
                
            catch ME
                fprintf('   !! 失败警告: 组 %d/样本 %d: %s !!\n', g, k, ME.message);
            end
            
            % 清理内存，防止图形窗口和变量堆积导致 MATLAB 崩溃 [cite: 125]
            clearvars -except base_dir groups num_groups sims_per_group g r_target c_target group_dir k start_g start_k
            close all
        end
    end
    fprintf('\n✅ 恭喜！所有 9 个生理组（共 9,000 次）仿真执行完毕。\n');
end