function BatchRun_Save_1000(N, base_dir, Fs_save, prefix, K_R, K_C, K_L)
if nargin<1, N=1000; end
if nargin<2 || strcmp(base_dir,'auto')
    base_dir=fullfile('step_one','dataset');
    if ~exist(base_dir,'dir'), mkdir(base_dir); end
end
if nargin<3, Fs_save=200; end
if nargin<4, prefix='FZ_'; end
if nargin<5, K_R=4; end
if nargin<6, K_C=3; end
if nargin<7, K_L=0; end
if ~exist(base_dir,'dir'), mkdir(base_dir); end
for k=1:N
    try
        rng(20260323+k,'twister');
        % 随机选择若干电阻/电容/电感进行倍率扰动
        R_mult=ones(1,25);
        C_mult=ones(1,21);
        L_mult=ones(1,3);
        R_candidates=setdiff(1:25,[11 17]); % 排除 Rsap(11)、Rvc(17)
        C_candidates=1:21;
        L_candidates=1:3;
        r_sel=R_candidates(randperm(numel(R_candidates),min(K_R,numel(R_candidates))));
        c_sel=C_candidates(randperm(numel(C_candidates),min(K_C,numel(C_candidates))));
        if K_L>0
            l_sel=L_candidates(randperm(numel(L_candidates),min(K_L,numel(L_candidates))));
        else
            l_sel=[];
        end
        for j=r_sel
            R_mult(j)=exp(log(0.85)+rand*(log(1.2)-log(0.85)));
        end
        for j=c_sel
            C_mult(j)=exp(log(0.9)+rand*(log(1.15)-log(0.9)));
        end
        for j=l_sel
            L_mult(j)=exp(log(0.9)+rand*(log(1.15)-log(0.9)));
        end
        Fcon_scalar=0.4+0.2*rand;
        Fvaso_scalar=0.4+0.2*rand;
        ok=false; tries=0;
        while ~ok && tries<100
            FhrS_scalar=0.3+0.4*rand;
            FhrV_scalar=0.3+0.4*rand;
            Hr=35+140*FhrS_scalar-40*FhrS_scalar.^2-32*FhrV_scalar+10*FhrV_scalar.^2-20*FhrV_scalar.*FhrS_scalar;
            ok=(Hr>=50)&&(Hr<=110);
            tries=tries+1;
        end
        % 变量在当前函数工作区定义，run 脚本时可见
        % R_mult, C_mult, L_mult, FhrV_scalar, FhrS_scalar, Fcon_scalar, Fvaso_scalar
        run('Entry_for_NormalHumanCirculationSystem.m');
        sid=sprintf('%s%04d',prefix,k);
        SaveTwoCycleSample(sid, base_dir, Fs_save);
        fprintf('---------------第%d次仿真成功------------------\n',k);
    catch ME
        fprintf('===============第%d次仿真失败：%s===============\n',k,ME.message);
    end
    clearvars -except N base_dir Fs_save prefix K_R K_C K_L k
    close all
end
end
