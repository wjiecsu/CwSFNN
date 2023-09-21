function [yPredTest,runtime]=SOFNN_ALA(Xtrain,Ytrain,Xtest,U,nMFs)
[Xtrain,Ytrain,Xtest,outputps] = NormilzeData(Xtrain, Ytrain,Xtest);
[M,N]=size(Xtrain);
rand('seed',2); %������ӹ̶�
NTest=size(Xtest,2); %TestSamNum����������500

%% ��������
E0=0.000001; %Ŀ�����
NAth=0.05;%������ֵ
HSth=0.01;%�޼���ֵ

% �������һ�����ġ���ȡ�Ȩֵ
C=0.4+rand(M,nMFs); %��������������
Sigma=ones(M,nMFs); %������������
W=rands(nMFs,1);    %������������֮��Ȩֵ

%%  ��ģ
tic
TrainSamInCurrent=[];  % ʹ����������ȫ������
for iter=1:N
    disp(['[SOFNN_ALA: ]','complete Iteration ->',num2str(iter),' steps'])
    TrainSamInEvery   = Xtrain(:,iter);%����һ��һ������ TrainSamNum ������ѵ����ʽ
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
    %% ��ȡѵ������TrainSamIn��TrainSamNum
    [CurrentIndim,CurrentNum]=size(TrainSamInCurrent);
    for k=1:CurrentNum
        %       k=1
        SamIn=TrainSamInCurrent(:,k);
        % ���������㣬ģ����
        for i=1:M
            for j=1:nMFs
                MemFunUnitOut(i,j)=exp(-(SamIn(i)-C(i,j))^2/Sigma(i,j)^2);
            end
        end
        % �����
        RuleUnitOut=prod(MemFunUnitOut); %��������
        % ��һ����
        RuleUnitOutSum=sum(RuleUnitOut); %�����������
        fBar(k,:)=RuleUnitOut./RuleUnitOutSum; %��һ�������������֯����NormValue
        % �����
        NetOut=fBar(k,:)*W; %�������������������
        Error=Ytrain(:,k)-NetOut;%���=�������-����ʵ�����
        
        % �Ľ�LM����ѧϰ
        for j=1:nMFs
            %Ȩֵ������
            AmendW(j)=-RuleUnitOut(j)./RuleUnitOutSum;
        end
        %����������
        for i=1:M
            for j=1:nMFs
                AmendCenter(i,j)=-2*((W(j)-NetOut)/RuleUnitOutSum)*RuleUnitOut(j)*(SamIn(i)-C(i,j))/(Sigma(i,j)^2);
            end
        end
        AmendCenterReshape=reshape(AmendCenter',1,M*nMFs);
        % ���������
        for i=1:M
            for j=1:nMFs
                AmendWidth(i,j)=-2*((W(j)-NetOut)/RuleUnitOutSum)*RuleUnitOut(j)*(SamIn(i)-C(i,j))^2/(Sigma(i,j)^3);
            end
        end
        AmendWidthReshape=reshape(AmendWidth',1,M*nMFs);
        
        %����Jacobi
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
    
    % �������ġ���ȡ�Ȩֵ
    C=C-AmendCenter';
    Sigma=Sigma-AmendWidth';
    W=W-AmendW;
    
    %% ѵ������Ԥ��
    TrainMemFunUnitOut=[];
    for k=1:N
        SamIn=Xtrain(:,k);
        % ���������㣬ģ����
        for i=1:M
            for j=1:nMFs
                TrainMemFunUnitOut(i,j)=exp(-((SamIn(i)-C(i,j))^2)/(Sigma(i,j)^2));
            end
        end
        % �����
        TrainRuleUnitOut=prod(TrainMemFunUnitOut); %��������
        % �����
        TrainRuleUnitOutSum=sum(TrainRuleUnitOut); %�����������
        TrainRuleValue=TrainRuleUnitOut./TrainRuleUnitOutSum; %������һ�����������֯ʱRuleNum�Ǳ仯��
        TrainNetOut(k)=TrainRuleValue*W; %�������������������
    end
    
    % ѵ��RMSE
    TrainError=Ytrain-TrainNetOut;
    TrainRMSE(iter)=sqrt(sum(TrainError.^2)/N);
    if  TrainRMSE(iter)<E0,break,end
    
    %---------------------�ṹ����֯-------------------------------------------------------------------
    if   mod(iter,5)==0  %ÿ5��ִ��һ������֯
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
            %---------------------------����-------------------------------
            for j=1:nMFs
                Dis=[];
                for k=1:nMFs
                    Dis(k)=norm(C(:,j)-C(:,k));
                end
                delt(j)=max(Dis);
                NA(j)=log(1+10*(TrainRuleValue(j)*delt(j))); %NAΪ�����ڵ����Ԫ����
            end
            NA_ratio=NA/sum(NA);
            NA_ratioMAX=max(NA_ratio);
            if TrainRMSE(iter)>TrainRMSE(iter-1) && NA_ratioMAX>NAth
                C=[C 1/2*(C(:,in(1))+SamIn)]; %in(1)���Ƕ�����ֵ�������ڵ����
                Sigma=[Sigma 1+rand(M,1)];
                Wnew=Error/RuleUnitOutSum;
                W=[W;Wnew];
                nMFs=nMFs+1;
            end
            fprintf(2,['start the prune operation\n'])
            %------------------------�޼�----------------------------------
            if TrainRMSE(iter)< TrainRMSE(iter-1) && min(HS_Value)<HSth
                HS_Delet_Num=length(find(HS_Value<HSth));
                if HS_Delet_Num>0
                    C(:,in(HS_Num-HS_Delet_Num+1:end))=[];  %ɾ�����
                    Sigma(:,in(HS_Num-HS_Delet_Num+1:end))=[];
                    W(in(HS_Num-HS_Delet_Num+1:end))=[];
                    nMFs=nMFs-HS_Delet_Num;
                end
            end
        end
    end
end
runtime=toc;
%% ��������Ԥ��
tic;
for k=1:NTest
    TestSamInSingle=Xtest(:,k);
    % ���������㣬ģ����
    for i=1:M
        for j=1:nMFs
            TestMemFunUnitOut(i,j)=exp(-((TestSamInSingle(i)-C(i,j))^2)/(Sigma(i,j)^2));
        end
    end
    % �����
    TestRuleUnitOut=prod(TestMemFunUnitOut); %��������
    % �����
    TestRuleUnitOutSum=sum(TestRuleUnitOut); %�����������
    TestRuleValue=TestRuleUnitOut./TestRuleUnitOutSum; %������һ�����������֯ʱRuleNum�Ǳ仯��
    TestNetOut(k,1)=TestRuleValue*W; %�������������������
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
[Ytrain,outputps]= mapminmax(Ytrain,0,1);    %��һ���������
Xtest=mapminmax('apply',Xtest,inputps);  %���Լ����ݹ�һ��
end

function Ypred =InverseNormilzeData(Ypred,outputps)
Ypred=mapminmax('reverse',Ypred,outputps);%����һ��
end

function [U,S,V,in]=jacobi_svd(A)
% clc;clear all; close all;
%   A=rand(3,2);
%   [U3,S3,V3]=svd(A);
%%  ����Jacobi�㷨
TOL=1.e-7;         %�趨һ����С����������̫С�����������ٶ���
[m,n]=size(A);     %m=500,n=20
U=A;               %U=500*20
V=eye(n);          %V�ǵ�λ����20*20�����Խ���Ԫ��Ϊ1������Ϊ0
converge=TOL+1;    %���������������Ļ�����+1
while converge>TOL %convergeΪ1��TOL�Ƿǳ�С����
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

[singvals_descend,in]=sort(singvals,2,'descend'); %��singvalsÿһ�а���������
S=diag(singvals(in)); %singvals(in)��һ��������S��һ������singvals(in)�������Խ�����
V=V(:,in);            %V��20*20
r=rank(S);            %rΪS���ȣ�r=20
U=U(:,in);            %U��500*20
U1=U(:,1:r);          %U1��500*20
U2=null(U1');         %U2��500*480
U=[U1,U2];            %U��500*500
if m>n                %500>20
    S(n+1:m,:)=zeros(m-n,n); %��S�� �����������0
end
end
