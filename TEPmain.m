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
U=0.2;         % LM优化器阻尼系数参数
lr=0.001;      % EBP优化器步长参数
lambda=0.001;  % 正则化参数
PopNum=50;     % PSO优化器种群大小参数
Ratio_theta=0.01; % 总相关性阈值
corr_theta =0.005; % 相关性阈值

%----------------------对比-----------------------------------------------------------
[yPredTest1,runtime1]         = SOFNN_PSO(Xtrain,Ytrain,Xtest,MaxEpoch,PopNum,nMFs);
[yPredTest2,runtime2]         = TSKFNN_EBP(Xtrain,Ytrain,Xtest,MaxEpoch,lr,lambda,nMFs);
% [yPredTest3,runtime3]         = FWNN(Xtrain,Ytrain,Xtest,MaxEpoch,nMFs);
% [yPredTest4,runtime4]         = ANFIS(Xtrain,Ytrain,Xtest,MaxEpoch,nMFs);
[yPredTest5,runtime5]         = MBGD_RDA(Xtrain,Ytrain,Xtest,alpha,beta1,beta2,lambda,nMFs,MaxEpoch,Nbs);
[yPredTest6,runtime6]         = SOFNN_ALA(Xtrain,Ytrain,Xtest,U,nMFs);
[yPredTest7,runtime7]           = CWSOFNN(Xtrain,Ytrain,Xtest,alpha, beta1, beta2,lambda,nMFs,corr_theta,Ratio_theta,MaxEpoch);
R2Test=1-(sum((yPredTest7-Ytest).^2)/sum((Ytest-mean(Ytest)).^2));
RMSETest=sqrt((Ytest-yPredTest7)'*(Ytest-yPredTest7)/200);
runtime=[runtime1,runtime2,runtime3,runtime4,runtime5,runtime6,runtime7];
yPredTest=[yPredTest0,yPredTest1,yPredTest2,yPredTest3,yPredTest4,yPredTest5,yPredTest6];
NTest=length(yPredTest);RMSETest=[];R2Test=[];
for i=1:size(yPredTest,2)
MSE(i)=(Ytest-yPredTest(:,i))'*(Ytest-yPredTest(:,i))/NTest;
MAE(i)=sum(abs((Ytest-yPredTest(:,i))/NTest));
RMSETest(i)=sqrt((Ytest-yPredTest(:,i))'*(Ytest-yPredTest(:,i))/NTest);
R2Test(i)=1-(sum((yPredTest(:,i)-Ytest).^2)/sum((Ytest-mean(Ytest)).^2));    
end
Yall=[Ytest yPredTest];
csvwrite('./Result/Yall.csv',Yall);

RMSETest7=sqrt((Ytest-yPredTest7)'*(Ytest-yPredTest7)/200);
