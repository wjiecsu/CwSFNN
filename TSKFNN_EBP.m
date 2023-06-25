function [yPredTest,runtime]=TSKFNN_EBP(Xtrain,Ytrain,Xtest,MaxEpoch,lr,lambda,nMFs)
%% data info
[Xtrain,Ytrain,Xtest,outputps] =NormilzeData(Xtrain, Ytrain,Xtest);
rand('seed',3); %随机种子固定
[N,M]=size(Xtrain);
NTest=size(Xtest,1);

%% set variable
nRules=nMFs^M; 
C=zeros(M,nMFs);
Sigma=C;
W=zeros(nRules,M+1);
nMFsVec=nMFs*ones(M,1);

%% Initialization
for m=1:M
    C(m,:)=linspace(min(Xtrain(:,m)),max(Xtrain(:,m)),nMFs);
    Sigma(m,:)=std(Xtrain(:,m));
end
minSigma=min(Sigma(:));
mu=zeros(M,nMFs);
Epoch=1;
%% Iterative update
tic;
while Epoch<=MaxEpoch 
    disp(['[TSKFNN: ]','complete Epoch ->',num2str(Epoch),' steps'])
    for k=1:N
        deltaC=zeros(M,nMFs); deltaSigma=zeros(M,nMFs);  deltaW(:,1:M+1)=lambda*W; deltaW(:,1)=zeros(nRules,1);% 每个batch都会置为0
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
        yR=[1 Xtrain(k,:)]*W';
        ypred(k)=fBar*yR';
        
        % 误差
        Error(k)=(ypred(k)-Ytrain(k));
        
        % Compute delta
        for r=1:nRules
            temp=Error(k)*(yR(r)*sum(f(k,:))-f(k,:)*yR')/sum(f(k,:))^2*f(k,r);
            if ~isnan(temp) && abs(temp)<inf
                vec=idx2vec(r,nMFsVec);
                % delta of c, sigma, and b
                for m=1:M
                    deltaC(m,vec(m))=temp*(Xtrain(k,m)-C(m,vec(m)))/Sigma(m,vec(m))^2;
                    deltaSigma(m,vec(m))=temp*(Xtrain(k,m)-C(m,vec(m)))^2/Sigma(m,vec(m))^3;
                    deltaW(r,m+1)=Error(k)*fBar(r)*Xtrain(k,m);
                end
                % delta of b0
                deltaW(r,1)=Error(k)*fBar(r);
            end
        end
        % SGD 
        W=W-lr*deltaW;
        C=C-lr*deltaC;
        Sigma=Sigma-lr*deltaSigma;
    end
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
end
yR=[ones(NTest,1) Xtest]*W';
yPredTest=sum(f.*yR,2)./sum(f,2); % prediction
yPredTest =InverseNormilzeData(yPredTest,outputps);
end %%%===>结束

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
