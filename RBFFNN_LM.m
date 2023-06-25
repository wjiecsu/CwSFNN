function [yPredTest,runtime]=RBFFNN_LM(Xtrain,Ytrain,Xtest,MaxEpoch,U,lambda,nMFs)
%% data info
[Xtrain,Ytrain,Xtest,outputps] =NormilzeData(Xtrain, Ytrain,Xtest);
rand('seed',3); %随机种子固定
[N,M]=size(Xtrain);
NTest=size(Xtest,1);

%% set variable
nRules=nMFs^M;
C=zeros(M,nMFs);
Sigma=C;
W=rand(nRules,1);
nMFsVec=nMFs*ones(M,1);

%% Initialization
for m=1:M
    C(m,:)=linspace(min(Xtrain(:,m)),max(Xtrain(:,m)),nMFs);
    Sigma(m,:)=std(Xtrain(:,m));
end
minSigma=min(Sigma(:));
mu=zeros(M,nMFs);


%% Iterative update
tic;
Epoch=1;
while Epoch<=MaxEpoch
    disp(['[RBFFNN_LM:]','complete Epoch ->',num2str(Epoch),' steps'])
    H=zeros(2*M*nMFs+nRules,2*M*nMFs+nRules);%海塞阵
    Jac=zeros(1,2*M*nMFs+nRules);            %Jac行向量
    gd=zeros(1,2*M*nMFs+nRules);             %梯度行向量
    for k=1:N
        deltaC=zeros(M,nMFs); deltaSigma=zeros(M,nMFs);  deltaW=zeros(nRules,1);% 每个batch都会置为0
        f=ones(N,nRules); % firing level of rules
        for m=1:M % membership grades of MFs
            mu(m,:)=exp(-(Xtrain(k,m)-C(m,:)).^2./(2*Sigma(m,:).^2));
        end
        for r=1:nRules
            idsMFs=idx2vec(r,nMFsVec);%idx2vec函数
            for m=1:M
                f(k,r)=f(k,r)*mu(m,idsMFs(m));
            end
        end
        fBar=f(k,:)/sum(f(k,:));
        ypred(k)=fBar*W;
        
        % 误差
        Error(k)=(ypred(k)-Ytrain(k));
        
        % 计算delta
        for r=1:nRules
            temp=W(r)*(sum(f(k,:))-f(k,r))*f(k,r)/sum(f(k,:))^2;
            if ~isnan(temp) && abs(temp)<inf
                vec=idx2vec(r,nMFsVec);
                % delta of c, sigma, and b
                for m=1:M
                    deltaC(m,vec(m))=temp*(Xtrain(k,m)-C(m,vec(m)))/Sigma(m,vec(m))^2;
                    deltaSigma(m,vec(m))=temp*(Xtrain(k,m)-C(m,vec(m)))^2/Sigma(m,vec(m))^3;
                end
                deltaW(r)=fBar(r)+lambda*(W(r));
            end
        end
        % LM优化算法
        deltaCreshape=reshape(deltaC,1,M*nMFs);          % M*nMFs--->按行 reshape 1*M*nMFs
        deltaSigmareshape=reshape(deltaSigma,1,M*nMFs);% M*nMFs--->按行 reshape 1*M*nMFs
        deltaWreshape=reshape(deltaW,1,nRules);          % nRules*1--->1* nRules
        Jac=horzcat(deltaCreshape,deltaSigmareshape,deltaWreshape);% 1*(2*M*nMFs+nRules);
        H=H+Jac'*Jac;%过雅各比矩阵Jac对黑塞矩阵H进行拟合
        gd=gd+Jac*Error(k);
    end
    
    %做批量更新
    % U 阻尼系数  
    H_lm=H+(U*eye(size(Jac,2),size(Jac,2)));%保证是正定的，lambda较大 Step=(1/lambda)g
    Step=inv(H_lm)*gd';%基于二阶的梯度下降法 获取步长Step : 2*M*nMFs+nRules * 1
    StepC    =reshape(Step(1:M*nMFs),M,nMFs);
    StepSigma=reshape(Step(M*nMFs+1:2*M*nMFs),M,nMFs);
    StepW    =reshape(Step(2*M*nMFs+1:end),nRules,1);
    
    %梯度下降
    W=W-StepW;
    C=C-StepC;
    Sigma=Sigma-StepSigma;
    
 
    Epoch=Epoch+1;
end
runtime=toc;

%% Test RMSE
f=ones(NTest,nRules); % firing level of rules
for k=1:NTest
    for m=1:M % membership grades of MFs
        mu(m,:)=exp(-(Xtest(k,m)-C(m,:)).^2./(2*Sigma(m,:).^2));
    end
    
    for r=1:nRules % firing levels of rules
        idsMFs=idx2vec(r,nMFsVec);
        for m=1:M
            f(k,r)=f(k,r)*mu(m,idsMFs(m));
        end
    end
    fBar=f(k,:)/sum(f(k,:));
    yPredTest(k,1)=fBar*W;
end
yPredTest =InverseNormilzeData(yPredTest,outputps);
end %==>结束

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
[Xtrain,inputps]  = mapminmax(Xtrain,0,1);
[Ytrain,outputps] = mapminmax(Ytrain,0,1);   %归一化后的数据
Xtest=mapminmax('apply',Xtest,inputps);  %测试集数据归一化
%还原
Xtrain=Xtrain';
Ytrain=Ytrain';
Xtest =Xtest' ;
end

function Ypred =InverseNormilzeData(Ypred,outputps)
Ypred=mapminmax('reverse',Ypred,outputps);%反归一化
end
