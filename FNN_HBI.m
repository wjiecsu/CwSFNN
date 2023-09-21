clc
clear all;
close all;

load('HBdata.mat')
X=HBdata(:,[1,5,6,7,8,9,2,4,14]);
Y=HBdata(:,18);
trainInd=1:600;
testInd=601:size(X,1);

Xtrain  = X(trainInd,:);
Ytrain  = Y(trainInd,:);
Xtest   = X(testInd,:);
Ytest   = Y(testInd,:);

nMFs=2;        % number of MFs in each input domain
Nbs =100;      % batch size
MaxEpoch=100;  % Epoch number
alpha=.01; beta1=0.9; beta2=0.999; % AdamBounder 优化器参数
U1=0.01;U2=0.001;U3=0.045;% LM优化器阻尼系数参数
lr=0.001;      % EBP优化器步长参数
PopNum=50;     % PSO优化器种群大小参数
lambda=0.001;  % 正则化参数
corr_theta=0.1;Ratio_theta=0.1;

%% Running Test
[yPredTest1,runtime1]       = SOFNN_PSO(Xtrain,Ytrain,Xtest,MaxEpoch,PopNum,nMFs);
[yPredTest2,runtime2]       = RBFFRNN_LM(Xtrain,Ytrain,Xtest,MaxEpoch,U1,lambda,nMFs);
[yPredTest3,runtime3]       = GKFRNN_EBP(Xtrain,Ytrain,Xtest,MaxEpoch,lr,lambda,nMFs);
[yPredTest4,runtime4]       = RBFFNN_EBP(Xtrain,Ytrain,Xtest,MaxEpoch,lr,lambda,nMFs);
[yPredTest5,runtime5]       = RBFFNN_LM(Xtrain,Ytrain,Xtest,MaxEpoch,U2,lambda,nMFs);
[yPredTest6,runtime6]       = TSKFNN_EBP(Xtrain,Ytrain,Xtest,MaxEpoch,lr,lambda,nMFs);
[yPredTest7,runtime7]       = FWNN(Xtrain,Ytrain,Xtest,MaxEpoch,nMFs);
[yPredTest8,runtime8]       = ANFIS(Xtrain,Ytrain,Xtest,MaxEpoch,nMFs);
[yPredTest9,runtime9]       = MBGD_RDA(Xtrain,Ytrain,Xtest,alpha, beta1, beta2,lambda,nMFs,MaxEpoch,Nbs);
[yPredTest10,runtime10]     = SOFNN_ALA(Xtrain,Ytrain,Xtest,U3,nMFs);
[yPredTest11,runtime11]     = CWSOFNN(Xtrain,Ytrain,Xtest,alpha, beta1, beta2,lambda,nMFs,corr_theta,Ratio_theta,MaxEpoch);
 
%% test performance
runtime=[runtime1,runtime2,runtime3,runtime4,runtime5,runtime6,runtime7,runtime8,runtime9,runtime10,runtime11];
yPredTest=[yPredTest1,yPredTest2,yPredTest3,yPredTest4,yPredTest5,yPredTest6,yPredTest7,yPredTest8,yPredTest9,yPredTest10,yPredTest11];
NTest=length(yPredTest);RMSETest=[];R2Test=[];
for i=1:size(yPredTest,2)
MSE(i)=(Ytest-yPredTest(:,i))'*(Ytest-yPredTest(:,i))/NTest;
MAE(i)=sum(abs((Ytest-yPredTest(:,i))/NTest));
RMSETest(i)=sqrt((Ytest-yPredTest(:,i))'*(Ytest-yPredTest(:,i))/NTest);
R2Test(i)=1-(sum((yPredTest(:,i)-Ytest).^2)/sum((Ytest-mean(Ytest)).^2));    
end

disp(['[Total running time:]  ==> ',num2str(floor(sum(runtime)/3600)),'h:',...
     num2str(floor(mod(sum(runtime),3600)/60)),...
    'min:',num2str(ceil(mod(sum(runtime),60))),'s'])















