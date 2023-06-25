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
MaxEpoch=100;  % Epoch number
alpha=.01; beta1=0.9; beta2=0.999; % AdamBounder 优化器参数
U=0.2;         % LM优化器阻尼系数参数
lr=0.001;      % EBP优化器步长参数
lambda=0.001;  % 正则化参数

corr_thetaSet =[0.01,0.03,0.05,0.08,0.10];      % 相关性阈值
Ratio_thetaSet=linspace(0.01,0.1,10); % 总相关性阈值
for i=1:length(corr_thetaSet)
    corr_theta=corr_thetaSet(i);
    for j=1:length(Ratio_thetaSet)
        Ratio_theta=Ratio_thetaSet(j);
        [yPredTest(:,(i-1)*length(Ratio_thetaSet)+j),runtime(1,(i-1)*length(Ratio_thetaSet)+j)]= CWSOFNN(Xtrain,Ytrain,Xtest,alpha, beta1, beta2,lambda,nMFs,corr_theta,Ratio_theta,MaxEpoch);
    end
end
%----------------------Performance-----------------------------------------------------------
NTest=length(yPredTest);RMSETest=[];R2Test=[];
for i=1:size(yPredTest,2)
    MSE(i)=(Ytest-yPredTest(:,i))'*(Ytest-yPredTest(:,i))/NTest;
    MAE(i)=sum(abs((Ytest-yPredTest(:,i))/NTest));
    RMSETest(i)=sqrt((Ytest-yPredTest(:,i))'*(Ytest-yPredTest(:,i))/NTest);
    R2Test(i)=1-(sum((yPredTest(:,i)-Ytest).^2)/sum((Ytest-mean(Ytest)).^2));
end
R2Test=reshape(R2Test,10,5);
runtime=reshape(runtime,10,5);
csvwrite('./Result/HyPerR2Test.csv',R2Test);
csvwrite('./Result/HyPerRuntime.csv',runtime);

