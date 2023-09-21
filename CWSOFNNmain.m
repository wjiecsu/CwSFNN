clc
clear all;
close all;

load('TEP.mat')
trainInd=1:1800;
testInd=1801:size(TEP,1);
X=TEP(:,1:8);
Y=TEP(:,10);
Xtrain  = X(trainInd,:);
Ytrain  = Y(trainInd,:);
Xtest   = X(testInd,:);
Ytest   = Y(testInd,:);

nMFs=2;        % number of MFs in each input domain
Nbs =100;      % batch size
MaxEpoch=100;  % Epoch number
alpha=.01; beta1=0.9; beta2=0.999; % AdamBounder 优化器参数
lambda=0.001;  % 正则化参数
Ratio_theta=0.01; % 总相关性阈值
corr_theta =0.005; % 相关性阈值

[Xtrain,Ytrain,Xtest,outputps] =NormilzeData(Xtrain,Ytrain,Xtest);
rand('seed',3); %随机种子固定
[N,M]=size(Xtrain);
NTest=size(Xtest,1);


% Defination
nMFsVec=nMFs*ones(M,1);
nMFsInit=nMFs;
nRules=nMFs^M; % number of rules
C=zeros(M,nMFs);
Sigma=C;
W=zeros(nRules,M+1);
idMFsList=[];
for r=1:nRules
    idsMFs=idx2vec(r,nMFsVec); %idx2vec函数
    idMFsList=[idMFsList;idsMFs];
end

% Initialization
for m=1:M
    C(m,:)=linspace(min(Xtrain(:,m))+0.1,max(Xtrain(:,m))-0.1,nMFs);
    Sigma(m,:)=std(Xtrain(:,m));
end
minSigma=min(Sigma(:));
mC=0; vC=0; mW=0; mSigma=0; vSigma=0; vW=0;
shuffle=randperm(N);
Epoch=1;iter=1;
eps=1e-8;
Ruleset=[];
tic;
% Iteration
while Epoch<=MaxEpoch
    % 初始化为空
    mu=zeros(M,nMFs);
    deltaC=zeros(M,nMFs); deltaSigma=deltaC;  deltaW=lambda*W; deltaW(:,1)=0; % 每个Epoch都会置为0
    f=ones(N,nRules); % firing level of rules
    disp(['[CWSOFNN:] ','complete Epoch ->',num2str(Epoch),' steps'])
    
    fBarset=[];
    fset=[];
    RuleUnitOutSum=0;
    
    % 前向传播
    for i=1:N
        for m=1:M % membership grades of MFs
            mu(m,:)=exp(-(Xtrain(i,m)-C(m,:)).^2./(2*Sigma(m,:).^2));
        end
        for r=1:nRules
            idsMFs=idMFsList(r,:);
            for m=1:M
                f(i,r)=f(i,r)*mu(m,idsMFs(m));
            end
        end
        fBar=f(i,:)/(sum(f(i,:))+eps);
        fBarset=[fBarset;fBar];
        yR=[1 Xtrain(i,:)]*W';                      %Wx+b W:R*(m+1)
        ypred(i)=fBar*yR';                          %预测
        
        % 记录数值
        RuleUnitOutSum=RuleUnitOutSum+sum(fBar);
        
        % 计算梯度
        for r=1:nRules
            temp=(ypred(i)-Ytrain(i))*(yR(r)*sum(f(i,:))-f(i,:)*yR')/sum(f(i,:))^2*f(i,r);
            if ~isnan(temp) && abs(temp)<inf
                vec=idMFsList(r,:);
                %% delta of c, sigma, W, and b
                for m=1:M
                    deltaC(m,vec(m))=deltaC(m,vec(m))+temp*(Xtrain(i,m)-C(m,vec(m)))/Sigma(m,vec(m))^2;
                    deltaSigma(m,vec(m))=deltaSigma(m,vec(m))+temp*(Xtrain(i,m)-C(m,vec(m)))^2/Sigma(m,vec(m))^3;
                    deltaW(r,m+1)=deltaW(r,m+1)+(ypred(i)-Ytrain(i))*fBar(r)*Xtrain(i,m);
                end
                %% delta of b0
                deltaW(r,1)=deltaW(r,1)+(ypred(i)-Ytrain(i))*fBar(r);
            end
        end
    end
    
    % AdaBound
    lb=alpha*(1-1/((1-beta2)*iter+1));
    ub=alpha*(1+1/((1-beta2)*iter));
    mC=beta1*mC+(1-beta1)*deltaC;
    vC=beta2*vC+(1-beta2)*deltaC.^2;
    mCHat=mC/(1-beta1^iter);
    vCHat=vC/(1-beta2^iter);
    lrC=min(ub,max(lb,alpha./(sqrt(vCHat)+10^(-8))));
    C=C-lrC.*mCHat; %更新C
    
    mSigma=beta1*mSigma+(1-beta1)*deltaSigma;
    vSigma=beta2*vSigma+(1-beta2)*deltaSigma.^2;
    mSigmaHat=mSigma/(1-beta1^iter);
    vSigmaHat=vSigma/(1-beta2^iter);
    lrSigma=min(ub,max(lb,alpha./(sqrt(vSigmaHat)+10^(-8))));
    Sigma  =max(minSigma,Sigma-lrSigma.*mSigmaHat);  %更新Sigma
    
    mW=beta1*mW+(1-beta1)*deltaW;
    vW=beta2*vW+(1-beta2)*deltaW.^2;
    mWHat=mW/(1-beta1^iter);
    vWHat=vW/(1-beta2^iter);
    lrW=min(ub,max(lb,alpha./(sqrt(vWHat)+10^(-8))));
    W=W-lrW.*mWHat;
    
    %%%% 自组织网络
    pearson_corr=[];%开始 进行增长和剪枝策略
    for u=1:nRules
        pearson_corr(u)=corr(fBarset(:,u),Ytrain);%fBarset:N*R
    end
    pearson_sum(Epoch)=sum(abs(pearson_corr));
    
    %-------------------------------增长-----------------------------------
    if (pearson_sum(Epoch)<max(pearson_sum(1:Epoch-1)))&(Epoch~=MaxEpoch)
        fprintf(2,['start the grow operation\n'])
        [~,max_index]=max(abs(pearson_corr));
        nMFs=nMFs+1;                   % 增长,只对最大的值进行增长
        nRules=nRules+1;
        idsMFs=nMFs*ones(1,M);         % 新增的
        idMFsList=[idMFsList;idsMFs];
        max_idMFs=idMFsList(max_index,:);
        for k=1:m %就是对相关性最大的隐节点分裂
            if max_idMFs(k)  <=1
                newC(k,1)    =1/2*C(k,max_idMFs(k))+C(k,max_idMFs(k)); 
                newSigma(k,1)=1/2*Sigma(k,max_idMFs(k))+Sigma(k,max_idMFs(k));
            else
                newC(k,1)    =1/2*C(k,max_idMFs(k)-1)+1/2*C(k,max_idMFs(k)); 
                newSigma(k,1)=1/2*Sigma(k,max_idMFs(k)-1)+1/2*C(k,max_idMFs(k));
            end    
        end
        %更新参数
        C=[C,newC];
        Sigma=[Sigma,newSigma];
        Wadd=W(max_index,:);
        W=[W;Wadd];
        
        %更新梯度
        mC=[mC,zeros(m,1)];
        vC=[vC,zeros(m,1)];
        mSigma=[mSigma,zeros(m,1)];
        vSigma=[vSigma,zeros(m,1)];
        mW=[mW;zeros(1,m+1)];
        vW=[vW;zeros(1,m+1)];
    end
    
    %----------------------------剪枝策略-----------------------------------
    Ratio=sum(abs(pearson_corr(find(abs(pearson_corr)<corr_theta))))/pearson_sum(Epoch);
    if (Ratio<Ratio_theta)&(Ratio~=0)&(Epoch~=MaxEpoch)% 删减，对最小的值进行删除
        fprintf(2,['start the delete operation\n'])
        min_index=find(abs(pearson_corr)<corr_theta);
        Rules_index=1:nRules;
        Is_detelte=ismember(Rules_index,min_index);
        new_Rules_index=Rules_index(~Is_detelte);
        idMFsList=idMFsList(new_Rules_index,:);
        nRules=nRules-size(min_index,2);
        
        %更新参数
        W=W(new_Rules_index,:);

        
        %更新梯度
        mW=mW(new_Rules_index,:);
        vW=vW(new_Rules_index,:);
    end
    
    % 参数记录
    Ruleset=[Ruleset;nRules];
    RMSE(Epoch,1)=sqrt(sum((ypred'-Ytrain).^2)/N);
    
    % 迭代参数更新
    iter=iter+1;
    Epoch=Epoch+1;
end
runtime=toc;
%%%%%% Test yPred
f=ones(NTest,nRules); % firing level of rules
for i=1:NTest
    for m=1:M % membership grades of MFs
        mu(m,:)=exp(-(Xtest(i,m)-C(m,:)).^2./(2*Sigma(m,:).^2));
    end
    
    for r=1:nRules % firing levels of rules
        idsMFs=idMFsList(r,:);
        for m=1:M
            f(i,r)=f(i,r)*mu(m,idsMFs(m));
        end
    end
end
yR=[ones(NTest,1) Xtest]*W';
yPredTest=sum(f.*yR,2)./sum(f,2); % prediction
yPredTest =InverseNormilzeData(yPredTest,outputps);



function vec=idx2vec(idx,nMFs)
% Convert from a scalar index of the rule to a vector index of MFs
vec=zeros(1,length(nMFs));
prods=[1; cumprod(nMFs(end:-1:1))];
if idx>prods(end)
    error('Error: idx is larger than the number of rules.');
end
prev=0;
for i=1:length(nMFs)
    vec(i)=floor((idx-1-prev)/prods(end-i))+1;
    prev=prev+(vec(i)-1)*prods(end-i);
end
end

function [Xtrain,Ytrain,Xtest,outputps] =NormilzeData(Xtrain, Ytrain,Xtest)
Xtrain=Xtrain';
Ytrain=Ytrain';
Xtest=Xtest';
[Xtrain,inputps] = mapminmax(Xtrain,0,1);
[Ytrain,outputps]= mapminmax(Ytrain,0,1);    %归一化后的数据
Xtest=mapminmax('apply',Xtest,inputps);      %测试集数据归一化
% 还原
Xtrain=Xtrain';
Ytrain=Ytrain';
Xtest=Xtest';
end

function Ypred =InverseNormilzeData(Ypred,outputps)
Ypred=mapminmax('reverse',Ypred,outputps);%反归一化
end