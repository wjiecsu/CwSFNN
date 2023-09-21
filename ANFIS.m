function [yPredTest,runtime] = ANFIS(Xtrain,Ytrain,Xtest,MaxEpoch,nMFs)
[Xtrain,Ytrain,Xtest,outputps] =NormilzeData(Xtrain,Ytrain,Xtest);
tic;
genOpt = genfisOptions('GridPartition');
genOpt.NumMembershipFunctions = nMFs;
genOpt.InputMembershipFunctionType = 'gaussmf';
inFIS = genfis(Xtrain,Ytrain,genOpt);
opt = anfisOptions('InitialFIS',inFIS,'EpochNumber',MaxEpoch);
opt.DisplayANFISInformation = 1;%显示ANFIS训练信息
opt.DisplayErrorValues = 0;     %显示ANFIS目标训练误差
opt.DisplayStepSize = 0;        %显示训练步长
opt.DisplayFinalResults = 0;    %显示训练结果
[fis,~,~]=anfis([Xtrain,Ytrain],opt);
yPredTest=evalfis(fis,Xtest);
yPredTest =InverseNormilzeData(yPredTest,outputps);
runtime=toc;
end

function [Xtrain,Ytrain,Xtest,outputps] =NormilzeData(Xtrain, Ytrain,Xtest)
Xtrain=Xtrain';
Ytrain=Ytrain';
Xtest=Xtest';
[Xtrain,inputps] = mapminmax(Xtrain,0,1);
[Ytrain,outputps]= mapminmax(Ytrain,0,1);    %归一化后的数据
Xtest=mapminmax('apply',Xtest,inputps);      %测试集数据归一化
% 还原
Xtrain=Xtrain';
Ytrain=Ytrain';
Xtest=Xtest';
end

function Ypred =InverseNormilzeData(Ypred,outputps)
Ypred=mapminmax('reverse',Ypred,outputps);%反归一化
end


