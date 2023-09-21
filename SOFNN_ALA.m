function [yPredTest,runtime]=SOFNN_ALA(Xtrain,Ytrain,Xtest,U,nMFs)
[Xtrain,Ytrain,Xtest,outputps] = NormilzeData(Xtrain, Ytrain,Xtest);
[M,N]=size(Xtrain);
rand('seed',2); %随机种子固定
NTest=size(Xtest,2); %TestSamNum测试样本数500

%% 参数设置
E0=0.000001; %目标误差
NAth=0.05;%增长阈值
HSth=0.01;%修剪阈值

% 随机产生一组中心、宽度、权值
C=0.4+rand(M,nMFs); %隶属函数层中心
Sigma=ones(M,nMFs); %隶属函数层宽度
W=rands(nMFs,1);    %规则层与输出层之间权值

%%  建模
tic
TrainSamInCurrent=[];  % 使得所有样本全部进入
for iter=1:N
    disp(['[SOFNN_ALA: ]','complete Iteration ->',num2str(iter),' steps'])
    TrainSamInEvery   = Xtrain(:,iter);%样本一个一个进入 TrainSamNum 金字塔训练方式
    TrainSamInCurrent = [TrainSamInCurrent, TrainSamInEvery];
    fBar=[];
    HistoryError=[];
    AmendCenter=[];
    AmendWidth=[];
    AmendW=[];
    MemFunUnitOut=[];
    WeightNum=2*M*nMFs+nMFs;
    jac=zeros(N,WeightNum);
    Q=zeros(WeightNum,WeightNum);
    g=zeros(1,WeightNum);
    %% 提取训练样本TrainSamIn，TrainSamNum
    [CurrentIndim,CurrentNum]=size(TrainSamInCurrent);
    for k=1:CurrentNum
        %       k=1
        SamIn=TrainSamInCurrent(:,k);
        % 隶属函数层，模糊化
        for i=1:M
            for j=1:nMFs
                MemFunUnitOut(i,j)=exp(-(SamIn(i)-C(i,j))^2/Sigma(i,j)^2);
            end
        end
        % 规则层
        RuleUnitOut=prod(MemFunUnitOut); %规则层输出
        % 归一化层
        RuleUnitOutSum=sum(RuleUnitOut); %规则层输出求和
        fBar(k,:)=RuleUnitOut./RuleUnitOutSum; %归一化层输出，自组织调整NormValue
        % 输出层
        NetOut=fBar(k,:)*W; %输出层输出，即网络输出
        Error=Ytrain(:,k)-NetOut;%误差=期望输出-网络实际输出
        
        % 改进LM参数学习
        for j=1:nMFs
            %权值修正量
            AmendW(j)=-RuleUnitOut(j)./RuleUnitOutSum;
        end
        %中心修正量
        for i=1:M
            for j=1:nMFs
                AmendCenter(i,j)=-2*((W(j)-NetOut)/RuleUnitOutSum)*RuleUnitOut(j)*(SamIn(i)-C(i,j))/(Sigma(i,j)^2);
            end
        end
        AmendCenterReshape=reshape(AmendCenter',1,M*nMFs);
        % 宽度修正量
        for i=1:M
            for j=1:nMFs
                AmendWidth(i,j)=-2*((W(j)-NetOut)/RuleUnitOutSum)*RuleUnitOut(j)*(SamIn(i)-C(i,j))^2/(Sigma(i,j)^3);
            end
        end
        AmendWidthReshape=reshape(AmendWidth',1,M*nMFs);
        
        %计算Jacobi
        jac=horzcat(AmendW,AmendCenterReshape,AmendWidthReshape);
        q=jac'*jac;
        yeta=jac*Error;
        Q=Q+q;
        g=g+yeta;
        HistoryError=[HistoryError Error];
    end
    
    % Improve LM
    H_lm=Q+(U*eye(WeightNum,WeightNum));
    AmendAll=inv(H_lm)*g';
    AmendW=AmendAll(1:nMFs);
    AmendCenter=AmendAll(nMFs+1:nMFs+M*nMFs);
    AmendCenter=reshape(AmendCenter,nMFs,M);
    AmendWidth=AmendAll(nMFs+M*nMFs+1:end);
    AmendWidth=reshape(AmendWidth,nMFs,M);
    
    % 更新中心、宽度、权值
    C=C-AmendCenter';
    Sigma=Sigma-AmendWidth';
    W=W-AmendW;
    
    %% 训练样本预测
    TrainMemFunUnitOut=[];
    for k=1:N
        SamIn=Xtrain(:,k);
        % 隶属函数层，模糊化
        for i=1:M
            for j=1:nMFs
                TrainMemFunUnitOut(i,j)=exp(-((SamIn(i)-C(i,j))^2)/(Sigma(i,j)^2));
            end
        end
        % 规则层
        TrainRuleUnitOut=prod(TrainMemFunUnitOut); %规则层输出
        % 输出层
        TrainRuleUnitOutSum=sum(TrainRuleUnitOut); %规则层输出求和
        TrainRuleValue=TrainRuleUnitOut./TrainRuleUnitOutSum; %规则层归一化输出，自组织时RuleNum是变化的
        TrainNetOut(k)=TrainRuleValue*W; %输出层输出，即网络输出
    end
    
    % 训练RMSE
    TrainError=Ytrain-TrainNetOut;
    TrainRMSE(iter)=sqrt(sum(TrainError.^2)/N);
    if  TrainRMSE(iter)<E0,break,end
    
    %---------------------结构自组织-------------------------------------------------------------------
    if   mod(iter,5)==0  %每5步执行一次自组织
        if  CurrentNum>nMFs && nMFs>1
            [U1,S1,V1,in]=jacobi_svd(fBar);
            HS=diag(S1);
            HS_Num=length(HS);
            HS_Value=[];
            HST=sort(HS,'ascend');
            for k=1:HS_Num
                HS_Value(k,:)=sum(HST(1:k))/sum(HS);
            end
            fprintf(2,['start the grow operation\n'])
            %---------------------------增长-------------------------------
            for j=1:nMFs
                Dis=[];
                for k=1:nMFs
                    Dis(k)=norm(C(:,j)-C(:,k));
                end
                delt(j)=max(Dis);
                NA(j)=log(1+10*(TrainRuleValue(j)*delt(j))); %NA为规则层节点的神经元活性
            end
            NA_ratio=NA/sum(NA);
            NA_ratioMAX=max(NA_ratio);
            if TrainRMSE(iter)>TrainRMSE(iter-1) && NA_ratioMAX>NAth
                C=[C 1/2*(C(:,in(1))+SamIn)]; %in(1)就是对奇异值最大的隐节点分裂
                Sigma=[Sigma 1+rand(M,1)];
                Wnew=Error/RuleUnitOutSum;
                W=[W;Wnew];
                nMFs=nMFs+1;
            end
            fprintf(2,['start the prune operation\n'])
            %------------------------修剪----------------------------------
            if TrainRMSE(iter)< TrainRMSE(iter-1) && min(HS_Value)<HSth
                HS_Delet_Num=length(find(HS_Value<HSth));
                if HS_Delet_Num>0
                    C(:,in(HS_Num-HS_Delet_Num+1:end))=[];  %删除多个
                    Sigma(:,in(HS_Num-HS_Delet_Num+1:end))=[];
                    W(in(HS_Num-HS_Delet_Num+1:end))=[];
                    nMFs=nMFs-HS_Delet_Num;
                end
            end
        end
    end
end
runtime=toc;
%% 测试样本预测
tic;
for k=1:NTest
    TestSamInSingle=Xtest(:,k);
    % 隶属函数层，模糊化
    for i=1:M
        for j=1:nMFs
            TestMemFunUnitOut(i,j)=exp(-((TestSamInSingle(i)-C(i,j))^2)/(Sigma(i,j)^2));
        end
    end
    % 规则层
    TestRuleUnitOut=prod(TestMemFunUnitOut); %规则层输出
    % 输出层
    TestRuleUnitOutSum=sum(TestRuleUnitOut); %规则层输出求和
    TestRuleValue=TestRuleUnitOut./TestRuleUnitOutSum; %规则层归一化输出，自组织时RuleNum是变化的
    TestNetOut(k,1)=TestRuleValue*W; %输出层输出，即网络输出
    yPredTest =InverseNormilzeData(TestNetOut,outputps);
    Testtime=toc;
    save('SOFNN_ALATestTime.mat','Testtime')
end
end

function [Xtrain,Ytrain,Xtest,outputps] =NormilzeData(Xtrain, Ytrain,Xtest)
Xtrain=Xtrain';
Ytrain=Ytrain';
Xtest=Xtest';
[Xtrain,inputps] = mapminmax(Xtrain,0,1);
[Ytrain,outputps]= mapminmax(Ytrain,0,1);    %归一化后的数据
Xtest=mapminmax('apply',Xtest,inputps);  %测试集数据归一化
end

function Ypred =InverseNormilzeData(Ypred,outputps)
Ypred=mapminmax('reverse',Ypred,outputps);%反归一化
end

function [U,S,V,in]=jacobi_svd(A)
% clc;clear all; close all;
%   A=rand(3,2);
%   [U3,S3,V3]=svd(A);
%%  单边Jacobi算法
TOL=1.e-7;         %设定一个较小的误差，但不能太小，否则收敛速度慢
[m,n]=size(A);     %m=500,n=20
U=A;               %U=500*20
V=eye(n);          %V是单位矩阵，20*20，主对角线元素为1，其他为0
converge=TOL+1;    %收敛条件，在误差的基础上+1
while converge>TOL %converge为1，TOL是非常小的数
    converge=0;
    for j=2:n
        for i=1:j-1
            % compute [alpha gamma;gamma beta]=(i,j) submatrix of U'*U
            alpha=U(:,i)'*U(:,i); %might be more than 1 line
            beta=U(:,j)'*U(:,j); %might be more than 1 line
            gamma=U(:,i)'*U(:,j); %might be more than 1 line
            converge=max(converge,abs(gamma));
            % compute Jacobi rotation that diagonalizes
            % [alpha gamma;gamma beta]
            if gamma==0
                c=1;
                s=0;
            else
                zeta=(beta-alpha)/(2*gamma);
                if norm(zeta)==0
                    t=1/(abs(zeta)+sqrt(1+zeta^2));
                else
                    t=sign(zeta)/(abs(zeta)+sqrt(1+zeta^2));
                end
                c=1/sqrt(1+t^2);
                s=c*t;
            end
            % update columns i and j of U
            t=U(:,i);
            U(:,i)=c*t-s*U(:,j);
            U(:,j)=s*t+c*U(:,j);
            % update matrix V of right singular vectors
            t=V(:,i);
            V(:,i)=c*t-s*V(:,j);
            V(:,j)=s*t+c*V(:,j);
        end
    end
end
% the singular values are the norms of the columns of U % the left singular vectors are the normalized columns of U

for j=1:n
    singvals(j)=norm(U(:,j));
    U(:,j)=U(:,j)/singvals(j);
end

[singvals_descend,in]=sort(singvals,2,'descend'); %对singvals每一行按降序排序
S=diag(singvals(in)); %singvals(in)是一个向量，S是一个矩阵，singvals(in)放在主对角线上
V=V(:,in);            %V是20*20
r=rank(S);            %r为S的秩，r=20
U=U(:,in);            %U是500*20
U1=U(:,1:r);          %U1是500*20
U2=null(U1');         %U2是500*480
U=[U1,U2];            %U是500*500
if m>n                %500>20
    S(n+1:m,:)=zeros(m-n,n); %将S阵 下面的子阵补上0
end
end
