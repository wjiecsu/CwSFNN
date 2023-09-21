clc
clear all;
close all;

load('TEP_multivate.mat')
TEP=MultipleOutData;
trainInd=1:1800;
testInd=1801:size(TEP,1);
X=TEP(:,1:8);
Y=TEP(:,[10,12,14]);
Xtrain  = X(trainInd,:);
Ytrain  = Y(trainInd,:);
Xtest   = X(testInd,:);
Ytest   = Y(testInd,:);

nMFs=2;        % number of MFs in each input domain
MaxEpoch=100;  % Epoch number
alpha=.01; beta1=0.9; beta2=0.999; % AdamBounder 优化器参数
lambda=0.001;  % 正则化参数
Ratio_theta=0.02; % 总相关性阈值
corr_theta =0.01; % 相关性阈值

% runtime1=0;
% ypred1=[];
% for i=1:3
% [ypred,runtime] = CWSOFNN(Xtrain,Ytrain(:,i),Xtest,alpha, beta1, beta2,lambda,nMFs,corr_theta,Ratio_theta,MaxEpoch);
% runtime1=runtime1+runtime;
% ypred1=[ypred1,ypred];
% end


[Xtrain,Ytrain,Xtest,outputps] =NormilzeData(Xtrain, Ytrain,Xtest);

% t2=tic;
% model2=trainRVM(Xtrain,Ytrain);
% runtime2=toc(t2);
% 
% t3=tic;
% model3=trainLSSVR(Xtrain,Ytrain);
% runtime3=toc(t3);
% 
% t4=tic;
% model4=trainSKRRrbf(Xtrain,Ytrain);
% runtime4=toc(t4);


t5=tic;
model5=trainWGPR(Xtrain,Ytrain);
runtime5=toc(t5);

% t6= tic;
% model6=trainELM(Xtrain,Ytrain);
% runtime6=toc(t6);


% ypred2=testRVM(model2,Xtest);
% ypred3=testLSSVR(model3,Xtest);
% ypred4=testSKRRrbf(model4,Xtest);
ypred5=testWGPR(model5,Xtest);
% ypred6=testELM(model6,Xtest);

% ypred2=InverseNormilzeData(ypred2,outputps);
% ypred3=InverseNormilzeData(ypred3,outputps);
% ypred4=InverseNormilzeData(ypred4,outputps);
ypred5=InverseNormilzeData(ypred5,outputps);
% ypred6=InverseNormilzeData(ypred6,outputps);


SStot  = sum((Ytest - mean(Ytest)).^2);
% SSres1 = sum((Ytest - ypred1).^2);
% SSres2 = sum((Ytest - ypred2).^2);
% SSres3 = sum((Ytest - ypred3).^2);
% SSres4 = sum((Ytest - ypred4).^2);
SSres5 = sum((Ytest - ypred5).^2);
% SSres6 = sum((Ytest - ypred6).^2);

% R2 评价指标
% R12 = 1 - SSres1/SStot;
% R22 = 1 - SSres2/SStot;
% R32 = 1 - SSres3/SStot;
% R42 = 1 - SSres4/SStot;
R52 = 1 - SSres5/SStot;
% R62 = 1 - SSres6/SStot;
% R2=[R12,R22,R32,R42,R52,R62];
% 
% % Runtime 评价指标
% Runtime=[runtime1,runtime2,runtime3,runtime4,runtime5,runtime6];



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
Ypred=Ypred';
Ypred=mapminmax('reverse',Ypred,outputps);%反归一化
Ypred=Ypred';
end
