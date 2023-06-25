function [yPredTest,runtime]=FWNN(Xtrain,Ytrain,Xtest,MaxEpoch,NMFs)
%%%主函数接口
%%%Layout
%%%1.根据输入变量维度产生第二层（模糊前规则）
%%%2.根据模糊前规则的组合生成模糊规则（全排列）
%%%3.中间处理，在FWNN函数中完成
%%%4.根据输入数据生成对应的小波神经网络
%%%5.损失函数MSE
%%%6.各个训练参数矩阵的偏导数
%%%7.梯度算法(fminunc)

[Xtrain,Ytrain,Xtest,outputps] =NormilzeData(Xtrain, Ytrain,Xtest);
tic
%模糊前规则数目规定
feature_num=size(Xtrain,2);
m=NMFs^feature_num;
%初始化参数矩阵
[miu,sigma,b_mat,c_mat,w_mat]=Init_para(NMFs,feature_num,m);
%unroll初始化参数矩阵
x0=[miu(:);sigma(:);b_mat(:);c_mat(:);w_mat(:)];
MaxFunctionEvaluations=MaxEpoch*length(Xtrain);
%主循环
fun=@(theta)costFunction(Xtrain,Ytrain,NMFs,theta);
options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton',...
    'MaxFunctionEvaluations',MaxFunctionEvaluations);
[x,fval,exitflag,output] = fminunc(fun,x0,options);
runtime=toc;
%unroll训练后参数矩阵
miu=reshape(x(1:NMFs*feature_num),NMFs,feature_num);
sigma=reshape(x((NMFs*feature_num+1):2*NMFs*feature_num),NMFs,feature_num);
b_mat=reshape(x((2*NMFs*feature_num+1):(2*NMFs*feature_num+feature_num*m))...
    ,feature_num,m);
c_mat=reshape(x((2*NMFs*feature_num+feature_num*m+1):(2*NMFs*feature_num+2*feature_num*m))...
    ,feature_num,m);
w_mat=reshape(x((2*NMFs*feature_num+2*feature_num*m+1):(2*NMFs*feature_num+3*feature_num*m))...
    ,feature_num,m);

%根据训练的参数矩阵进行预测
fwmodel_out = Fwnnmodel(Xtest,zeros(length(Xtest),1), NMFs,miu,sigma,b_mat,c_mat,w_mat);%测试无需Y，zeros(length(Xtest),1)
predict=fwmodel_out.Y_est;
yPredTest =InverseNormilzeData(predict,outputps);
end

%% Fwnnmodel
function  [fwmodel] = Fwnnmodel(S, Y, pre,miu,sigma,b_mat,c_mat,w_mat)
%%%Layout
%%%1.根据输入变量维度产生第二层（模糊前规则）
%%%2.根据模糊前规则的组合生成模糊规则（全排列）
%%%3.中间处理，在主函数中完成
%%%4.根据输入数据生成对应的小波神经网络
%%%5.损失函数MSE
%%%6.各个训练参数的偏导数
%%%7.梯度算法

%输入矩阵纬度
[sample_num,feature_num]=size(S);
%检测输出矩阵纬度是否匹配
[m_out,out_dim]=size(Y);
if(m_out~=sample_num)
    A=input("数据不匹配");
end
if(out_dim~=1)
    B=input("输出仅限一维数据");
end
y_est=zeros(sample_num,1);
yita_all=zeros(sample_num,1);
yita=zeros(pre^feature_num,sample_num);
phai=zeros(pre^feature_num,sample_num);
index=zeros(pre^feature_num,feature_num);
for i=1:sample_num%每个数据挨个输入产生模糊网络
x=S(i,:);
%Layer2,产生模糊前规则,每一列对应一组前规则
A_fuzz=Gauss_Func(x,miu,sigma);%A_fuzz(pre*feature_num)
%Layer3,产生模糊规则共m条
m=pre^feature_num;
[fuzz_rule,index]=Fuzz_Rule(feature_num,pre,A_fuzz);%fuzz_rule=(pre^feature_num,1)
%Layer4,输出模糊网络的权重
yita_all(i)=sum(fuzz_rule(:));
yita(:,i)=fuzz_rule/yita_all(i);
%Layer5,共pre^feature_num个子小波神经网络
phai(:,i)=Wnn(x,feature_num,m,b_mat,c_mat,w_mat);%phai(m,1)
%Layer6,输出
y_est(i)=(yita(:,i)')*phai(:,i);
end
fwmodel=struct('Y_est',y_est,'Y_true',Y,'X',S,'Miu_mat',miu, ...
    'Sigma_mat',sigma,'B_mat',b_mat,'C_mat',c_mat, ...
    'W_mat',w_mat,'Yita',yita,'Yita_all',yita_all,'Pre',pre, ...
    'Phai',phai,'Index',index);
end

%% Init_para
function [miu,sigma,b_mat,c_mat,w_mat]=Init_para(pre,feature_num,m)
%暂用置1初始化，也可采用c-clusteting
miu=ones(pre,feature_num);
sigma=ones(pre,feature_num);
b_mat=rand(feature_num,m);
c_mat=rand(feature_num,m);
w_mat=rand(feature_num,m);
end

%% Wnn
function phai=Wnn(x,feature_num,m,b_mat,c_mat,w_mat)
x=repmat(x',1,m);
mid=((x-b_mat)./c_mat).^2;
phai=sum(w_mat.*((ones(feature_num,m)-mid).*(exp(-1/2*mid))),1);
phai=phai';
end

%% Fuzz_Rule
function [fuzz_rule,k]=Fuzz_Rule(feature_num,pre,A_fuzz)
m=pre^feature_num;
fuzz_rule=ones(m,1);
k=ones(m,feature_num);%初始化index矩阵
origin_k=ones(1,feature_num);
for ik=1:m%生成index矩阵
    add_k=zeros(1,feature_num);
    total=ik-1;
    for ik2=1:feature_num
        add_k(feature_num-ik2+1)=mod(total,pre);
        total=(total-mod(total,pre))/pre;
    end
    k(ik,:)=origin_k+add_k;
end
for i1=1:m
    for i2=1:feature_num
    fuzz_rule(i1)=fuzz_rule(i1)*A_fuzz(k(i1,i2),i2);
    end
end
end

%% costFunction
function[jval,gradientVec]=costFunction(S,Y,pre,theta)
%[miu,sigma,b_mat,c_mat,w_mat]
feature_num=size(S,2);
m=pre^feature_num;
miu=reshape(theta(1:pre*feature_num),pre,feature_num);
sigma=reshape(theta((pre*feature_num+1):2*pre*feature_num),pre,feature_num);
b_mat=reshape(theta((2*pre*feature_num+1):(2*pre*feature_num+feature_num*m))...
    ,feature_num,m);
c_mat=reshape(theta((2*pre*feature_num+feature_num*m+1):(2*pre*feature_num+2*feature_num*m))...
    ,feature_num,m);
w_mat=reshape(theta((2*pre*feature_num+2*feature_num*m+1):(2*pre*feature_num+3*feature_num*m))...
    ,feature_num,m);
[fwmodel] = Fwnnmodel(S, Y, pre,miu,sigma,b_mat,c_mat,w_mat);
jval=1/size(fwmodel.X,1)*sum((fwmodel.Y_true-fwmodel.Y_est).^2);
if nargout>1
    gradientVec=Gradient_form(fmodel);
end
end

%% Gaussian function
function A=Gauss_Func(x,miu,sigma)
pre_in=size(sigma,1);
x=repmat(x,pre_in,1);
% miu=repmat(miu,1,size(x,2));
% sigma=repmat(sigma,1,size(x,2));
A=exp(-1/2*((x-miu)./sigma).^2);
end

%% Gradient_form
function [gradient]=Gradient_form(fmodel)
[sample_num,feature_num]=size(fmodel.X);
m=(fmodel.Pre)^feature_num;
%pE_y=2/size(fmodel.X,1)*sum(fmodel.Y_est-fmodel.Y_true);
one_mat=ones(feature_num,m);
pE_y_inside=zeros(sample_num,1);
pE_w=zeros(feature_num,m);
pE_b=zeros(feature_num,m);
pE_c=zeros(feature_num,m);
pE_miu=zeros(pre,feature_num);
pE_sigma=zeros(pre,feature_num);
for i=1:sample_num
    pE_y_inside(i)=2*(fmodel.Y_est(i)-fmodel.Y_true(i));
    x=fmodel.X(i,:);
    x=repmat(x',1,m);
    mid_w=((x-fmodel.B_mat)./fmodel.C_mat).^2;
    yita=repmat((fmodel.Yita(:,i))',feature_num,1);
    %E对w偏导(feature_num,m)
    pE_w=pE_w+pE_y_inside(i)*yita.*...
        ((one_mat-mid_w).*(exp(-1/2*mid_w)));
    %E对b偏导(feature_num,m)
    mid_b=((x-fmodel.B_mat)./fmodel.C_mat).^2;
    pE_b=pE_b+pE_y_inside(i)*yita.*fmodel.W_mat...
        .*((x-fmodel.B_mat)./(fmodel.C_mat.^2)).*...
        (3*one_mat-mid_b).*exp(-1/2*mid_b);
    %E对c偏导(feature_num,m)
    mid_c=((x-fmodel.B_mat).^2)./((fmodel.C_mat).^3);
    pE_c=pE_c+pE_y_inside(i)*yita.*fmodel.W_mat...
        .*mid_c.*(3*one_mat-mid_b).*exp(-1/2*mid_b);
    %y对A偏导(pre,feature_num)
    p_A_inside=Partial_a(fmodel.X(i,:),fmodel.Miu,...
        fmodel.Sigma,feature_num,pre,fmodel.Index,...
        fmodle.Phai(:,i),fmodel.Yita_all(i,1));
    %E对miu偏导(pre,feature_num)
    mid_miu=((repmat(fmodel.X(i,:),pre,1)-fmodel.Miu)./...
        fmodel.Sigma).^2;
    pE_miu=pE_miu+pE_y_inside(i)*p_A_inside.*...
        (repmat(fmodel.X(i,:),pre,1)-fmodel.Miu)./...
        ((fmodel.Sigma.^2)).*exp(-1/2*mid_miu);
    %E对sigma偏导(pre,feature_num)
    pE_sigma=pE_sigma+pE_y_inside(i)*p_A_inside.*...
        ((repmat(fmodel.X(i,:),pre,1)-fmodel.Miu).^2)./...
        ((fmodel.Sigma.^3)).*exp(-1/2*mid_miu);
end
    pE_w=pE_w/sample_num;
    pE_b=pE_b/sample_num;
    pE_c=pE_c/sample_num;
    pE_miu=pE_miu/sample_num;
    pE_sigma=pE_sigma/sample_num;
% gradient=struct('pE_w',pE_w,'pE_b',pE_b,'pE_c',pE_c...
%     ,'pE_miu',pE_miu,'pE_sigma',pE_sigma);
gradient=[pE_miu(:);pE_sigma(:);pE_b(:);pE_b(:);pE_w(:)];
end

%% Partial_a
function p_A_inside=Partial_a(x,y,miu,sigma,feature_num,...
    pre,index,phai,yita_all)
    %从y对A1,i1偏导
    p_A_inside=zeros(pre,feature_num);
    A_fuzz=Gauss_Func(x,miu,sigma);
for i=1:feature_num
    A_fuzz_d=A_fuzz;
    A_fuzz_d(:,i)=[];
    index(:,[1,i])=index(:,[i,1]);
    for j=1:pre
    %对非第一列元素去除相应列，非第一行元素对应index值为j
    [sub_fuzz,k]=Fuzz_Rule(feature_num-1,pre,A_fuzz_d);
    k_add=[j*ones(size(sub_fuzz,1)),k];
    up_1=0;
    up_2=0;
    for i1=1:size(k_add,1)
        for i2=1:size(index,1)
            if(isequal(k_add(i1,:),index(i2,:)))
                up_1=up_1+sub_fuzz(i1)*phai(i2);
                up_2=up_2+y*sub_fuzz(i1);
            end 
        end
    end
    p_A_inside(i,j)=(up_1-up_2)/yita_all;
    end
    index(:,[1,i])=index(:,[i,1]);
end
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