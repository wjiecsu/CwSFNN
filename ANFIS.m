function [yPredTest,runtime] = ANFIS(Xtrain,Ytrain,Xtest,MaxEpoch,nMFs)
[Xtrain,Ytrain,Xtest,outputps] =NormilzeData(Xtrain,Ytrain,Xtest);
tic;
genOpt = genfisOptions('GridPartition');
genOpt.NumMembershipFunctions = nMFs;
genOpt.InputMembershipFunctionType = 'gaussmf';
inFIS = genfis(Xtrain,Ytrain,genOpt);
opt = anfisOptions('InitialFIS',inFIS,'EpochNumber',MaxEpoch);
opt.DisplayANFISInformation = 1;%��ʾANFISѵ����Ϣ
opt.DisplayErrorValues = 0;     %��ʾANFISĿ��ѵ�����
opt.DisplayStepSize = 0;        %��ʾѵ������
opt.DisplayFinalResults = 0;    %��ʾѵ�����
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
[Ytrain,outputps]= mapminmax(Ytrain,0,1);    %��һ���������
Xtest=mapminmax('apply',Xtest,inputps);      %���Լ����ݹ�һ��
% ��ԭ
Xtrain=Xtrain';
Ytrain=Ytrain';
Xtest=Xtest';
end

function Ypred =InverseNormilzeData(Ypred,outputps)
Ypred=mapminmax('reverse',Ypred,outputps);%����һ��
end


