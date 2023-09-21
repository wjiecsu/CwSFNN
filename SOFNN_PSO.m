function [yPredTest,runtime]=SOFNN_PSO(Xtrain,Ytrain,Xtest,MaxEpoch,PopNum,nMFs)
%% data info
[Xtrain,Ytrain,Xtest,outputps] =NormilzeData(Xtrain, Ytrain,Xtest);
rand('seed',3); %随机种子固定
[N,m]=size(Xtrain); %InDim输入维数4，TrainSamNum训练样本数500
nRules=nMFs^m;

%% set variable
%nMFs=2;
%PopNum = 50; %种群规模
%MaxEpoch=200;

%% 优化器参数设置
pop_Rule = round((nRules-1)*rand(PopNum,1)+1);   % 每一个个体所携带的模糊规则数，PopNum行1列,1-15之间的随机整数
pop_dim = m*nMFs;
pop_bound_center = [0.1  1];    % 中心范围
pop_bound_width =  [0.1  1];    % 宽度范围

c1=1.49445;
c2=1.49445;
w=0.729;
Max_FES=PopNum*MaxEpoch;
xLower1=pop_bound_center(1);
xLower2=pop_bound_width(1);
xTop1=pop_bound_center(2);
xTop2=pop_bound_width(2);
Vmin1 = -0.5*(xTop1-xLower1);
Vmin2 = -0.5*(xTop2-xLower2);
Vmax1 = -Vmin1;
Vmax2 = -Vmin2;

%% 种群初始化
v=Vmin1+2.*Vmax1.*rand(PopNum,2*pop_dim);
fecount=0;
pop=zeros(PopNum,2*pop_dim);
for i=1:PopNum
    pop(i,1:m*nMFs)         = pop_bound_center(1)+rand(pop_dim,1)*(pop_bound_center(2)-pop_bound_center(1));  %中心，前几行
    pop(i,m*nMFs+1:2*m*nMFs)= pop_bound_width(1)+ rand(pop_dim,1)*(pop_bound_width(2)-pop_bound_width(1));    %宽度，后几行
end

%% 初始值
for i=1:PopNum
    fit(i)= fitness(pop(i,:),pop_Rule(i),Xtrain,Ytrain);
end
pbest=pop;
fitnesspbest=fit; %initialize the pbest and the pbest's fitness value
[fitnessgbest,ix]=min(fitnesspbest);
gbest=pbest(ix,:);%initialize the gbest and the gbest's fitness value

%% 记录器定义
gbestMatrix=repmat(gbest,PopNum,1);
gbestrecord=zeros(MaxEpoch,2*pop_dim);
fitnessgbestrecord=zeros(MaxEpoch,1);
fitnessgbestrecord(1)=fitnessgbest;
gbestrecord(1,:)=gbest;
RuleNumbest=0;
weightbest=[];
tic;
%开始迭代
for iter=1:MaxEpoch
    disp(['[SOFNN_PSO:]','complete Iteration ->',num2str(iter),' steps'])
    for i=1:PopNum
        v(i,:)=w.*v(i,:)+c1.*rand(1,2*pop_dim).*(pbest(i,:)-pop(i,:))+c2*rand(1,2*pop_dim).*(gbestMatrix(i,:)-pop(i,:));
        v(i,find(v(i,1:pop_dim)>Vmax1))=Vmax1;
        v(i,pop_dim+find(v(i,pop_dim+1:2*pop_dim)>Vmax2))=Vmax2;
        v(i,find(v(i,1:pop_dim)<Vmin1))=Vmin1;
        v(i,pop_dim+find(v(i,pop_dim+1:2*pop_dim)<Vmin2))=Vmin2;
        pop(i,:)=pop(i,:)+v(i,:);
        pop(i,find(pop(i,1:pop_dim)>xTop1))=xTop1;
        pop(i,pop_dim+find(pop(i,pop_dim+1:2*pop_dim)>xTop2))=xTop2;
        pop(i,find(pop(i,1:pop_dim)<xLower1))=xLower1;
        pop(i,pop_dim+find(pop(i,pop_dim+1:2*pop_dim)<xLower2))=xLower2;
        [fit(i),weight]=fitness(pop(i,:),pop_Rule(i),Xtrain,Ytrain);
        fecount=fecount+1;
        if fit(i)<fitnesspbest(i)
            pbest(i,:)=pop(i,:);
            fitnesspbest(i)=fit(i);
        end
        if fitnesspbest(i)<fitnessgbest
            gbest       =pbest(i,:);     %最优位置
            weightbest  =weight;         %最优权重
            RuleNumbest =pop_Rule(i);    %最优规则数
            fitnessgbest=fitnesspbest(i);%最优值
            gbestMatrix=repmat(gbest,PopNum,1);%update the gbest
        end
    end
    gbestrecord(iter,:)=gbest;
    fitnessgbestrecord(iter)=fitnessgbest;
    if fecount>=Max_FES
        break;
    end
end
runtime=toc;

%% 训练集预测
C=[];Sigma=[];fBar=[];W=[];
C=gbest(1:pop_dim);              % 前4行元素是中心
Sigma=gbest(pop_dim+1:2*pop_dim);        % 后4行元素是宽度
W=weightbest;
fBar=GetMeNormValue(Xtest,C,Sigma,RuleNumbest,nMFs);
ypred=fBar*W;
yPredTest=mapminmax('reverse',ypred,outputps);%反归一化
end  %%%===>结束



function [fit,Weights]=fitness(pop,RuleNum,Xtrain,Ytrain)
% 输入
[N,M]=size(Xtrain); %输入维数、训练样本数
OutDim=size(Ytrain,1);   %输出维数
nMFs=2;
Center=[];Width=[];fBar=[];Weights=[];
Center=pop(1:M*nMFs);               % 前4行元素是中心
Width=pop(M*nMFs+1:2*M*nMFs);         % 后4行元素是宽度
fBar=GetMeNormValue(Xtrain,Center,Width,RuleNum,nMFs);
Weights=DeriveWeights(fBar,Ytrain);
NetOut=fBar*Weights;
RMSE=sqrt(sumsqr(Ytrain-NetOut)/(OutDim*N)); %训练样本的RMSE
fit=RMSE*(1+0.9*RuleNum); %均方根误差就是适应度值
end

function NormValue=GetMeNormValue(Xtrain,C,Sigma,nRules,nMFs)
[N,M]=size(Xtrain); %InDim是输入维数，SamNum是样本个数
C=reshape(C,M,nMFs);
Sigma=reshape(Sigma,M,nMFs);
nMFsVec=nMFs*ones(M,1);
Rules=1:nRules;
fBar=zeros(N,nRules);
f=ones(N,nRules); % firing level of rules
for k=1:N
    for m=1:M % membership grades of MFs
        mu(m,:)=exp(-((Xtrain(k,m)-C(m,:)).^2)./(2*Sigma(m,:).^2));
        %！！！ C一定要在定义域内否则容易产生NAN
    end
    for j=1:nRules
        idsMFs=idx2vec(Rules(j),nMFsVec);%idx2vec函数
        for m=1:M
            f(k,j)=f(k,j)*mu(m,idsMFs(m));
            if(mu(m,idsMFs(m))==0)
                disp('C不在定义域！！！')
            end
        end
    end
    fBar(k,:)=f(k,:)/sum(f(k,:));
end
NormValue=fBar;
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

% 采用最小二乘获取输出权重
function Weights=DeriveWeights(fBar,Ytrain)
Q = pinv(fBar'*fBar);
Weights=Q*fBar'*Ytrain;
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

