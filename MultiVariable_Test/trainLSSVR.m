function model= trainLSSVR(Xtrain,Ytrain)
model = initlssvm(Xtrain,Ytrain,'f',[],[],'RBF_kernel','original');
% 'f' 代表回归，'c'代表分类
model = tunelssvm(model,'simplex','leaveoneoutlssvm',{'mse'});
%crossvalidatelssvm 交叉验证法
model = trainlssvm(model);
end

