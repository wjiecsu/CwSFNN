function model= trainLSSVR(Xtrain,Ytrain)
model = initlssvm(Xtrain,Ytrain,'f',[],[],'RBF_kernel','original');
% 'f' ����ع飬'c'�������
model = tunelssvm(model,'simplex','leaveoneoutlssvm',{'mse'});
%crossvalidatelssvm ������֤��
model = trainlssvm(model);
end

