function [yPredTest,runtime]=GKFRNN_EBP(Xtrain,Ytrain,Xtest,MaxEpoch,lr,lambda,nMFs)
%% Hyper-parameters
% MaxEpoch=100;
% lambda  =0.01;
% nMFs    =2;
% lr      =0.001;
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
pref=ones(1,nRules);
beta=rand(1,nRules);
mu=zeros(M,nMFs);

%% Initialization
param.c=2; % 指定聚类个数
param.e=1e-6; % 设定允许偏差
param.ro=ones(1,param.c); %1行4列，全为1
result = GKcluster(Xtrain,param); %执行GK聚类
C=result.cluster.v';
for m=1:M
    Sigma(m,:)=std(Xtrain(:,m));
end

%% Iterative update
tic;
Epoch=1;
while Epoch<=MaxEpoch
    disp(['[GKFRNN_EBP]','complete Epoch ->',num2str(Epoch),' steps'])
    for k=1:N
        deltaC=zeros(M,nMFs); deltaSigma=zeros(M,nMFs);  deltaW=zeros(nRules,1); deltabeta=zeros(1,nRules);% 每个batch都会置为0% 每个batch都会置为0
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
        
        forgetf  =beta.*pref;
        Recurout=1./(1+exp(-forgetf));
        f(k,:)=f(k,:).*Recurout;
        fBar=f(k,:)/sum(f(k,:));    
        ypred(k)=fBar*W;
        
        % 误差
        Error(k)=(ypred(k)-Ytrain(k)); %预测值减去真实值，预测值在前
        
        % Compute delta
        for r=1:nRules
            temp=2*Error(k)*W(r)*(sum(f(k,:))-f(k,r))*f(k,r)/sum(f(k,:))^2;
            if ~isnan(temp) && abs(temp)<inf
                vec=idx2vec(r,nMFsVec);
                % delta of c, sigma, and b
                for m=1:M
                    deltaC(m,vec(m))=temp*(Xtrain(k,m)-C(m,vec(m)))/Sigma(m,vec(m))^2;
                    deltaSigma(m,vec(m))=temp*(Xtrain(k,m)-C(m,vec(m)))^2/Sigma(m,vec(m))^3;
                end
                deltaW(r)=Error(k)*fBar(r)+lambda*(W(r)^2);
                deltabeta(r)=temp*(exp(-forgetf(r))./(1+exp(-forgetf(r)))^2);
            end
        end
        % SGD
        W=W-lr*deltaW;
        C=C-lr*deltaC;
        Sigma=Sigma-lr*deltaSigma;
        beta=beta-lr*deltabeta;
        pref=f(k,:);
    end
    Epoch=Epoch+1;
end
runtime=toc;

%% Test RMSE
f=ones(NTest,nRules); % firing level of rules
Testpref=ones(1,nRules);
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
    Recurout=1./(1+exp(-beta.*Testpref));
    f(k,:)=f(k,:).*Recurout;  
    fBar=f(k,:)/sum(f(k,:));
    yPredTest(k,1)=fBar*W;
    
    %递归调用
    Testpref=f(k,:);
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
end%%%===>结束

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

%% GK 聚类函数
function result = GKcluster(data,param)

%checking the parameters given  %检查给定参数
f0=param.c;  %param.c=4;
X=data;

[N,n] = size(X);
[Nf0,nf0] = size(f0);
X1 = ones(N,1);
%default parameters    %默认参数
if exist('param.m')==1, m = param.m;else m = 2;end;
if exist('param.e')==1, e = param.m;else e = 1e-4;end;
if exist('param.ro')==1, rho=param.ro;
else 
    if max(Nf0,nf0) == 1
        rho = ones(1,param.c);
    else
        rho = ones(1,size(f0,2));
    end
end
if exist('param.gamma')==1, gamma = param.gamma;else gamma = 0;end;
if exist('param.beta')==1, beta = param.beta;else beta = 1e15;end;


% Initialize fuzzy partition matrix   %初始化模糊划分矩阵
rand('state',0)
if max(Nf0,nf0) == 1		% only number of cluster given
  c = f0;
  mm = mean(X);
  aa = max(abs(X - ones(N,1)*mm));
  v = 2*(ones(c,1)*aa).*(rand(c,n)-0.5) + ones(c,1)*mm; 
  for j = 1 : c
    xv = X - X1*v(j,:);
    d(:,j) = sum((xv.^2),2);
  end
  d = (d+1e-10).^(-1/(m-1));
  f0 = (d ./ (sum(d,2)*ones(1,c)));
else        %if the partition matrix was given
  c = size(f0,2);
  fm = f0.^m; sumf = sum(fm);
  v = (fm'*X)./(sumf'*ones(1,n));
end

f = zeros(N,c);                 % partition matrix
iter = 0;                       % iteration counter
A0= eye(n)*det(cov(X)).^(1/n);  % "identity" matr.


% Iterate
while  max(max(f0-f)) > e
  iter = iter + 1;
  f = f0;
  % Calculate centers
  fm = f.^m; sumf = sum(fm);
  v = (fm'*X)./(sumf'*ones(1,n));
  for j = 1 : c
    xv = X - X1*v(j,:);
    % Calculate covariance matrix for each clusters
    A = ones(n,1)*fm(:,j)'.*xv'*xv/sumf(j);
    %Covariance estimating for the GK algorithm
    A =(1-gamma)*A+gamma*(A0.^(1/n));
    if cond(A)> beta
        [ev,ed]=eig(A); edmax = max(diag(ed));
        ed(beta*ed < edmax) = edmax/beta;
        A = ev*diag(diag(ed))*inv(ev);
    end
    %Calculate distances
    M = (1/det(pinv(A))/rho(j))^(1/n)*pinv(A);
    %M(:,:,j) = (det(A)/rho(j)).^(1/n)*pinv(A);
    d(:,j) = sum((xv*M.*xv),2);
  end
    
  distout=sqrt(d);
  
  J(iter) = sum(sum(f0.*d));           %calculate objective function
  % Update f0
  d = (d+1e-10).^(-1/(m-1));
  f0 = (d ./ (sum(d,2)*ones(1,c)));
   
end             %end of iteration period

fm = f.^m; sumf = sum(fm);

P = zeros(n,n,c);             % covariance matrix
M = P;                          % norm-inducing matrix
V = zeros(c,n);                % eigenvectors
D = V;                          % eigenvalues

% calculate P,V,D,M
for j = 1 : c                        
    xv = X - ones(N,1)*v(j,:);
    % Calculate covariance matrix
    A = ones(n,1)*fm(:,j)'.*xv'*xv/sumf(j);
    % Calculate eigen values and eigen vectors
    [ev,ed] = eig(A); ed = diag(ed)';
    ev = ev(:,ed == min(ed));
    % Put cluster info in one matrix
	P(:,:,j) = A;
    M(:,:,j) = (det(A)/rho(j)).^(1/n)*pinv(A);
    V(j,:) = ev';
    D(j,:) = ed;
end

%result outputs
result.data.f = f0;
result.data.d = distout;
result.cluster.v = v;
result.cluster.P = P;
result.cluster.M = M;
result.cluster.V = V;
result.cluster.D = D;
result.iter = iter;
result.cost = J; %J为目标函数
end
