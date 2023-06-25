function [yPredTest,runtime]=FNN_EBP(Xtrain,Ytrain,Xtest,MaxEpoch,nMFs)
[Xtrain,Ytrain,Xtest,outputps] = NormilzeData(Xtrain, Ytrain,Xtest);
[M,N]=size(Xtrain);
rand('seed',4); %随机种子固定
NTest=size(Xtest,2); %TestSamNum测试样本数500

%% 参数设置
E0=0.0000001; %目标误差
lr=0.01; %学习率,取0.01（1000步）；0.1（300步）

%随机产生一组中心、宽度、权值，归一化后都取rand
C=rand(M,nMFs); %隶属函数层中心
Sigma=ones(M,nMFs); %隶属函数层宽度,取ones很重要
W=rand(nMFs,1); %规则层与输出层之间权值

%%  建模
tic
%% MaxEpoch
for epoch=1:MaxEpoch
    disp(['complete Epochs ->',num2str(epoch),' steps'])
    for k=1:N
        AmendW=0*W;
        AmendWidth=0*Sigma;
        AmendCenter=0*C;
        SamIn=Xtrain(:,k);
        % 隶属函数层，模糊化
        for i=1:M
            for j=1:nMFs
                MemFunUnitOut(i,j)=exp(-(SamIn(i)-C(i,j))^2/Sigma(i,j)^2);
            end
        end
        % 规则层
        RuleUnitOut=prod(MemFunUnitOut,1);% 规则层输出
        % 归一化层
        RuleUnitOutSum=sum(RuleUnitOut);  % 规则层输出求和
        fBar=RuleUnitOut./RuleUnitOutSum; % 归一化层输出，自组织调整NormValue
        % 输出层
        NetOut=fBar*W;                    % 输出层输出，即网络输出
        Error(k)=Ytrain(:,k)-NetOut;      % 误差=期望输出-网络实际输出  e=yd-y
        
        % 梯度
        % 权值修正量   
        for j=1:nMFs
            AmendW(j)=-Error(k)*fBar(j);
        end
        %中心修正量      
        for i=1:M
            for j=1:nMFs
                AmendCenter(i,j)=-Error(k)*W(j)*(RuleUnitOutSum-RuleUnitOut(j))*RuleUnitOut(j)*2*(SamIn(i)-C(i,j))/(Sigma(i,j)^2*RuleUnitOutSum^2);
            end
        end
        % 宽度修正量      
        for i=1:M
            for j=1:nMFs
                AmendWidth(i,j)=-Error(k)*W(j)*(RuleUnitOutSum-RuleUnitOut(j))*RuleUnitOut(j)*2*(SamIn(i)-C(i,j))^2/(Sigma(i,j)^3*RuleUnitOutSum^2);
            end
        end
        % 更新中心、宽度、权值
        W=W-lr*AmendW;
        C=C-lr*AmendCenter;
        Sigma=Sigma-lr*AmendWidth;
    end
    % 训练RMSE
    RMSE(epoch)=sqrt(sum(Error.^2)/N); %N样本个数
    if RMSE(epoch)<E0,break,end
end
runtime=toc;

%% 测试样本预测
for k=1:NTest
    SamIn=Xtest(:,k);
    % 隶属函数层，模糊化
    for i=1:M
        for j=1:nMFs
            TestMemFunUnitOut(i,j)=exp(-((SamIn(i)-C(i,j))^2)/(Sigma(i,j)^2));
        end
    end
    % 规则层
    TestRuleUnitOut=prod(TestMemFunUnitOut); %规则层输出
    % 输出层
    TestRuleUnitOutSum=sum(TestRuleUnitOut); %规则层输出求和
    TestRuleValue=TestRuleUnitOut./TestRuleUnitOutSum; %规则层归一化输出，自组织时RuleNum是变化的
    TestNetOut(k,1)=TestRuleValue*W; %输出层输出，即网络输出
    yPredTest =InverseNormilzeData(TestNetOut,outputps);
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
