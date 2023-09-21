function [yPredTest,runtime]=MBGD_RDA(Xtrain,Ytrain,Xtest,alpha, beta1, beta2,lambda,nMFs,MaxEpoch,Nbs)
tic;
[Xtrain,Ytrain,Xtest,outputps] =NormilzeData(Xtrain, Ytrain,Xtest);
rand('seed',3); %随机种子固定
[N,M]=size(Xtrain);
NTest=size(Xtest,1);
if Nbs>N
    Nbs=N;
end
nMFsVec=nMFs*ones(M,1);
nRules=nMFs^M; % number of rules
C=zeros(M,nMFs);
Sigma=C;
W=zeros(nRules,M+1);
% Initialization
for m=1:M
    C(m,:)=linspace(min(Xtrain(:,m)),max(Xtrain(:,m)),nMFs);
    Sigma(m,:)=std(Xtrain(:,m));
end
minSigma=min(Sigma(:));

%% Iterative update
mu=zeros(M,nMFs);
mC=0; vC=0; mW=0; mSigma=0; vSigma=0; vW=0; 
shuffle=randperm(N);
Epoch=1;batch=1;iter=1;
while Epoch<=MaxEpoch
    deltaC=zeros(M,nMFs); deltaSigma=deltaC;  deltaW=lambda*W; deltaW(:,1)=0; % 每个batch都会置为0
    f=ones(Nbs,nRules); % firing level of rules
    
    %随机采样 或者 %idsTrain=datasample(1:N,Nbs,'replace',false);
    if batch*Nbs<=N
        disp(['[MBGD_RDA: ]','complete Batch ->',num2str(batch)])
        idsTrain=shuffle((batch-1)*Nbs+1:batch*Nbs);
    else
        disp(['complete Epoch ->',num2str(Epoch),' steps']) 
        idsTrain=shuffle(N-Nbs+1:N);
        batch=0;shuffle=randperm(N);
        Epoch=Epoch+1;
    end
    for n=1:Nbs
        for m=1:M % membership grades of MFs
            mu(m,:)=exp(-(Xtrain(idsTrain(n),m)-C(m,:)).^2./(2*Sigma(m,:).^2));
        end
        for r=1:nRules
            idsMFs=idx2vec(r,nMFsVec);%idx2vec函数
            for m=1:M
                f(n,r)=f(n,r)*mu(m,idsMFs(m));
            end
        end
        fBar=f(n,:)/sum(f(n,:));
        yR=[1 Xtrain(idsTrain(n),:)]*W';
        ypred(n)=fBar*yR'; % prediction
        
        % Compute delta
        for r=1:nRules
            temp=(ypred(n)-Ytrain(idsTrain(n)))*(yR(r)*sum(f(n,:))-f(n,:)*yR')/sum(f(n,:))^2*f(n,r);
            if ~isnan(temp) && abs(temp)<inf
                vec=idx2vec(r,nMFsVec);
                % delta of c, sigma, and b
                for m=1:M
                    deltaC(m,vec(m))=deltaC(m,vec(m))+temp*(Xtrain(idsTrain(n),m)-C(m,vec(m)))/Sigma(m,vec(m))^2;
                    deltaSigma(m,vec(m))=deltaSigma(m,vec(m))+temp*(Xtrain(idsTrain(n),m)-C(m,vec(m)))^2/Sigma(m,vec(m))^3;
                    deltaW(r,m+1)=deltaW(r,m+1)+(ypred(n)-Ytrain(idsTrain(n)))*fBar(r)*Xtrain(idsTrain(n),m);
                end
                % delta of b0
                deltaW(r,1)=deltaW(r,1)+(ypred(n)-Ytrain(idsTrain(n)))*fBar(r);
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
    C=C-lrC.*mCHat;
    
    mSigma=beta1*mSigma+(1-beta1)*deltaSigma;
    vSigma=beta2*vSigma+(1-beta2)*deltaSigma.^2;
    mSigmaHat=mSigma/(1-beta1^iter);
    vSigmaHat=vSigma/(1-beta2^iter);
    lrSigma=min(ub,max(lb,alpha./(sqrt(vSigmaHat)+10^(-8))));
    Sigma=max(minSigma,Sigma-lrSigma.*mSigmaHat);
    
    mW=beta1*mW+(1-beta1)*deltaW;
    vW=beta2*vW+(1-beta2)*deltaW.^2;
    mWHat=mW/(1-beta1^iter);
    vWHat=vW/(1-beta2^iter);
    lrW=min(ub,max(lb,alpha./(sqrt(vWHat)+10^(-8))));
    W=W-lrW.*mWHat;
    
    %iteration
    batch=batch+1;
    iter=iter+1;
end
runtime=toc;
% Test RMSE
tic;
f=ones(NTest,nRules); % firing level of rules
for n=1:NTest
    for m=1:M % membership grades of MFs
        mu(m,:)=exp(-(Xtest(n,m)-C(m,:)).^2./(2*Sigma(m,:).^2));
    end
    
    for r=1:nRules % firing levels of rules
        idsMFs=idx2vec(r,nMFsVec);
        for m=1:M
            f(n,r)=f(n,r)*mu(m,idsMFs(m));
        end
    end
end
yR=[ones(NTest,1) Xtest]*W';
yPredTest=sum(f.*yR,2)./sum(f,2); % prediction
yPredTest =InverseNormilzeData(yPredTest,outputps);
Testtime=toc;
save('MBGD_RDATestTime.mat','Testtime')
end

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
[Xtrain,inputps]  = mapminmax(Xtrain);
[Ytrain,outputps] = mapminmax(Ytrain);   %归一化后的数据
Xtest=mapminmax('apply',Xtest,inputps);  %测试集数据归一化
%还原
Xtrain=Xtrain';
Ytrain=Ytrain';
Xtest =Xtest' ;
end

function Ypred =InverseNormilzeData(Ypred,outputps)
Ypred=mapminmax('reverse',Ypred,outputps);%反归一化
end
