function SaveTwoCycleSample(sample_id, base_dir, Fs_save)
if nargin<2, base_dir='dataset'; end
if nargin<3, Fs_save=200; end
step=evalin('caller','step');
t=evalin('caller','t');
allP=evalin('caller','allP');
allV=evalin('caller','allV');
allQ=evalin('caller','allQ');
allD=evalin('caller','allD');
HrT=evalin('caller','HrT');
Tall=evalin('caller','Tall');
FhrV=evalin('caller','FhrV');
FhrS=evalin('caller','FhrS');
Fcon=evalin('caller','Fcon');
Fvaso=evalin('caller','Fvaso');
try R=evalin('caller','R'); catch, R=[]; end
try C=evalin('caller','C'); catch, C=[]; end
try Rc=evalin('caller','Rc'); catch, Rc=[]; end
try L=evalin('caller','L'); catch, L=[]; end
try yinit=evalin('caller','yinit'); catch, yinit=[]; end
HrT=HrT(HrT~=0);
Tedges=[0 cumsum(HrT)];
center_t=450;
if center_t>=Tedges(end)
    center_t=Tedges(max(1,end-1))+0.001;
end
k=find(Tedges<=center_t,1,'last');
if isempty(k), k=2; end
ks=max(2,min(k,numel(Tedges)-1));
ts=Tedges(ks-1);
te=Tedges(ks+1);
is=max(1,floor(ts/step)+1);
ie=min(numel(t),floor(te/step)+1);
twin=t(is:ie);
Pwin=allP(is:ie,:);
Vwin=allV(is:ie,:);
Qwin=allQ(is:ie,:);
Dwin=allD(is:ie,:);
Fs_orig=1/step;
M=max(1,round(Fs_orig/Fs_save));
idx=1:M:numel(twin);
t_out=twin(idx)-twin(idx(1));
P_out=Pwin(idx,:);
V_out=Vwin(idx,:);
Q_out=Qwin(idx,:);
D_out=Dwin(idx,:);
Hr_mean=NaN;
try
    Hr=evalin('caller','Hr');
    Hr_mean=mean(Hr(max(1,floor(is):min(numel(Hr),ie))));
catch
    Hr_mean=60/mean(HrT(max(1,ks-5):min(numel(HrT),ks+5)));
end
Hr_var=0;
try
    Hr=evalin('caller','Hr');
    Hr_var=var(Hr(max(1,floor(is):min(numel(Hr),ie))));
catch
    Hr_var=0;
end
stable=true;
try
    LV=Vwin(:,1);
    RV=Vwin(:,18);
    m1=max(LV)-min(LV);
    m2=max(RV)-min(RV);
    stable=isfinite(m1)&&isfinite(m2)&&m1>0&&m2>0;
catch
    stable=true;
end
chP={'Plv','Phaa','Plna','Plca','Paop','Prula','Prica','Plica','Plula','Psap','Prsv','Prijv','Plijv','Plsv','Psv','Pvc','Pra','Prv','Prpap','Plpap','Prpad','Plpad','Prpv','Plpv','Pla'};
chV={'Vlv','Vhaa','Vlna','Vlca','Vaop','Vrula','Vrica','Vlica','Vlula','Vsap','Vrsv','Vrijv','Vlijv','Vlsv','Vsv','Vvc','Vra','Vrv','Vrpap','Vlpap','Vrpad','Vlpad','Vrpv','Vlpv','Vla'};
chQ={'Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q71','Q72','Q77','Q8','Q9','Q10','Q11','Q12','Q13','Q14','Q15','Q16','Q17','Q18','Q19','Q20','Q21','Q22','Q23','Q240','Q250','Q24','Q25','Q26','Q27','Q28','Q29','Q30'};
chD={'Da','Dm','Dp','Dt','D1','D2','D3','D4','D51','D52','D53','D54','D6','D7','D8','D9','D10','D11','D12'};
out_dir=fullfile(base_dir,'runs',sample_id);
if ~exist(base_dir,'dir'), mkdir(base_dir); end
if ~exist(fullfile(base_dir,'runs'),'dir'), mkdir(fullfile(base_dir,'runs')); end
if ~exist(out_dir,'dir'), mkdir(out_dir); end
inp.FhrV=FhrV(1); inp.FhrS=FhrS(1); inp.Fcon=Fcon(1); inp.Fvaso=Fvaso(1);
inp.R=R; inp.C=C; inp.Rc=Rc; inp.L=L; inp.yinit=yinit;
inp.Tall=Tall; inp.step=step; inp.Fs_save=Fs_save; inp.M=M; inp.fc_Hz=0.4*Fs_save;
inp.center_t=center_t;
fid=fopen(fullfile(out_dir,'inputs.json'),'w');
fwrite(fid,jsonencode(inp),'char'); fclose(fid);
wavefile=fullfile(out_dir,'waveform.mat');
channelsP=chP; channelsV=chV; channelsQ=chQ; channelsD=chD;
Fs_save=Fs_save; Fs_orig=Fs_orig; fc_Hz=0.4*Fs_save; ts=ts; te=te; cycles=[ks-1 ks];
P=P_out; V=V_out; Q=Q_out; D=D_out; t=t_out; M=M; Hr_mean=Hr_mean; Hr_var=Hr_var; stable=stable;
save(wavefile,'t','P','V','Q','D','channelsP','channelsV','channelsQ','channelsD','Fs_orig','Fs_save','M','fc_Hz','ts','te','cycles','Hr_mean','Hr_var','stable','-v7.3');
beat1_start=Tedges(ks-1); beat1_end=Tedges(ks);
beat2_start=Tedges(ks); beat2_end=Tedges(ks+1);
i1s=max(1,floor(beat1_start/step)+1); i1e=min(numel(t),floor(beat1_end/step)+1);
i2s=max(1,floor(beat2_start/step)+1); i2e=min(numel(t),floor(beat2_end/step)+1);
Vlv1=allV(i1s:i1e,1); Vlv2=allV(i2s:i2e,1);
Vrv1=allV(i1s:i1e,18); Vrv2=allV(i2s:i2e,18);
LV_EDV1=max(Vlv1); LV_ESV1=min(Vlv1); LV_SV1=LV_EDV1-LV_ESV1; LV_EF1=LV_SV1/max(LV_EDV1,eps);
LV_EDV2=max(Vlv2); LV_ESV2=min(Vlv2); LV_SV2=LV_EDV2-LV_ESV2; LV_EF2=LV_SV2/max(LV_EDV2,eps);
RV_EDV1=max(Vrv1); RV_ESV1=min(Vrv1); RV_SV1=RV_EDV1-RV_ESV1; RV_EF1=RV_SV1/max(RV_EDV1,eps);
RV_EDV2=max(Vrv2); RV_ESV2=min(Vrv2); RV_SV2=RV_EDV2-RV_ESV2; RV_EF2=RV_SV2/max(RV_EDV2,eps);
Paop1=allP(i1s:i1e,5); Paop2=allP(i2s:i2e,5);
P_AOP_sys1=max(Paop1); P_AOP_dia1=min(Paop1); P_AOP_mean1=mean(Paop1); PP_AOP1=P_AOP_sys1-P_AOP_dia1;
P_AOP_sys2=max(Paop2); P_AOP_dia2=min(Paop2); P_AOP_mean2=mean(Paop2); PP_AOP2=P_AOP_sys2-P_AOP_dia2;
Pvc1=allP(i1s:i1e,16); Pvc2=allP(i2s:i2e,16);
P_VC_mean1=mean(Pvc1); P_VC_mean2=mean(Pvc2);
Ppap1=allP(i1s:i1e,19); Ppap2=allP(i2s:i2e,19);
sPAP1=max(Ppap1); dPAP1=min(Ppap1); mPAP1=(sPAP1+2*dPAP1)/3;
sPAP2=max(Ppap2); dPAP2=min(Ppap2); mPAP2=(sPAP2+2*dPAP2)/3;
Q2_1=allQ(i1s:i1e,2); Q2_2=allQ(i2s:i2e,2);
Q21_1=allQ(i1s:i1e,24); Q21_2=allQ(i2s:i2e,24);
Q23_1=allQ(i1s:i1e,26); Q23_2=allQ(i2s:i2e,26);
Q25_1=allQ(i1s:i1e,30); Q25_2=allQ(i2s:i2e,30);
Q29_1=allQ(i1s:i1e,34); Q29_2=allQ(i2s:i2e,34);
Q2_peak=max([max(Q2_1) max(Q2_2)]); Q2_mean=mean([mean(Q2_1) mean(Q2_2)]);
Q21_peak=max([max(Q21_1) max(Q21_2)]); Q21_mean=mean([mean(Q21_1) mean(Q21_2)]);
Q23_peak=max([max(Q23_1) max(Q23_2)]); Q23_mean=mean([mean(Q23_1) mean(Q23_2)]);
Q25_peak=max([max(Q25_1) max(Q25_2)]); Q25_mean=mean([mean(Q25_1) mean(Q25_2)]);
Q29_peak=max([max(Q29_1) max(Q29_2)]); Q29_mean=mean([mean(Q29_1) mean(Q29_2)]);
Dseg=Dwin;
Dm_on_frac=mean(Dseg(:,2));
Da_on_frac=mean(Dseg(:,1));
Dt_on_frac=mean(Dseg(:,4));
Dp_on_frac=mean(Dseg(:,3));
totV1=sum(allV(i1s:i1e,:),2); totV2=sum(allV(i2s:i2e,:),2);
totalV_drift=(mean(totV2)-mean(totV1))/max(mean([totV1;totV2]),eps);
if numel(Vlv1)>=3 && numel(Paop1)>=3
    PV_area1=abs(polyarea(Vlv1,Paop1));
else
    PV_area1=NaN;
end
if numel(Vlv2)>=3 && numel(Paop2)>=3
    PV_area2=abs(polyarea(Vlv2,Paop2));
else
    PV_area2=NaN;
end
feat.sample_id = string(sample_id);
feat.Hr_mean = Hr_mean;
feat.V_LV_EDV1 = LV_EDV1; feat.V_LV_ESV1 = LV_ESV1; feat.SV_LV1 = LV_SV1; feat.EF_LV1 = LV_EF1;
feat.V_LV_EDV2 = LV_EDV2; feat.V_LV_ESV2 = LV_ESV2; feat.SV_LV2 = LV_SV2; feat.EF_LV2 = LV_EF2;
feat.V_RV_EDV1 = RV_EDV1; feat.V_RV_ESV1 = RV_ESV1; feat.SV_RV1 = RV_SV1; feat.EF_RV1 = RV_EF1;
feat.V_RV_EDV2 = RV_EDV2; feat.V_RV_ESV2 = RV_ESV2; feat.SV_RV2 = RV_SV2; feat.EF_RV2 = RV_EF2;
feat.P_AOP_sys1 = P_AOP_sys1; feat.P_AOP_dia1 = P_AOP_dia1; feat.P_AOP_mean1 = P_AOP_mean1; feat.PP_AOP1 = PP_AOP1;
feat.P_AOP_sys2 = P_AOP_sys2; feat.P_AOP_dia2 = P_AOP_dia2; feat.P_AOP_mean2 = P_AOP_mean2; feat.PP_AOP2 = PP_AOP2;
feat.sPAP1 = sPAP1; feat.dPAP1 = dPAP1; feat.mPAP1 = mPAP1;
feat.sPAP2 = sPAP2; feat.dPAP2 = dPAP2; feat.mPAP2 = mPAP2;
feat.P_VC_mean1 = P_VC_mean1; feat.P_VC_mean2 = P_VC_mean2;
feat.Q2_peak = Q2_peak; feat.Q2_mean = Q2_mean;
feat.Q21_peak = Q21_peak; feat.Q21_mean = Q21_mean;
feat.Q23_peak = Q23_peak; feat.Q23_mean = Q23_mean;
feat.Q25_peak = Q25_peak; feat.Q25_mean = Q25_mean;
feat.Q29_peak = Q29_peak; feat.Q29_mean = Q29_mean;
feat.Dm_on_frac = Dm_on_frac; feat.Da_on_frac = Da_on_frac; feat.Dt_on_frac = Dt_on_frac; feat.Dp_on_frac = Dp_on_frac;
feat.totalV_drift = totalV_drift;
feat.PV_LV_area1 = PV_area1; feat.PV_LV_area2 = PV_area2;
feat.stable = logical(stable);
T = struct2table(feat,'AsArray',true);
featfile=fullfile(out_dir,'features.csv');
writetable(T,featfile);
idxfile=fullfile(base_dir,'index.csv');
IndexRow=table(string(sample_id),string(strrep(wavefile,'\','/')),string(strrep(featfile,'\','/')),...
    Hr_mean,ts,te,string(['[' num2str(ks-1) ',' num2str(ks) ']']),true,...
    'VariableNames',{'sample_id','waveform_path','features_path','Hr_mean','ts','te','cycles','stable'});
try
    if exist(idxfile,'file')
        try
            writetable(IndexRow,idxfile,'WriteMode','append');
        catch
            IT=readtable(idxfile,'TextType','string');
            IT=[IT; IndexRow];
            writetable(IT,idxfile);
        end
    else
        writetable(IndexRow,idxfile);
    end
catch
    bdir=fullfile(base_dir,'index_pending');
    if ~exist(bdir,'dir'), mkdir(bdir); end
    writetable(IndexRow,fullfile(bdir,[sample_id '.csv']));
end
end
