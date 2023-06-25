function [yPredTest,runtime]=FWNN(Xtrain,Ytrain,Xtest,MaxEpoch,NMFs)
%%%�������ӿ�
%%%Layout
%%%1.�����������ά�Ȳ����ڶ��㣨ģ��ǰ����
%%%2.����ģ��ǰ������������ģ������ȫ���У�
%%%3.�м䴦����FWNN���������
%%%4.���������������ɶ�Ӧ��С��������
%%%5.��ʧ����MSE
%%%6.����ѵ�����������ƫ����
%%%7.�ݶ��㷨(fminunc)

[Xtrain,Ytrain,Xtest,outputps] =NormilzeData(Xtrain, Ytrain,Xtest);
tic
%ģ��ǰ������Ŀ�涨
feature_num=size(Xtrain,2);
m=NMFs^feature_num;
%��ʼ����������
[miu,sigma,b_mat,c_mat,w_mat]=Init_para(NMFs,feature_num,m);
%unroll��ʼ����������
x0=[miu(:);sigma(:);b_mat(:);c_mat(:);w_mat(:)];
MaxFunctionEvaluations=MaxEpoch*length(Xtrain);
%��ѭ��
fun=@(theta)costFunction(Xtrain,Ytrain,NMFs,theta);
options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton',...
    'MaxFunctionEvaluations',MaxFunctionEvaluations);
[x,fval,exitflag,output] = fminunc(fun,x0,options);
runtime=toc;
%unrollѵ�����������
miu=reshape(x(1:NMFs*feature_num),NMFs,feature_num);
sigma=reshape(x((NMFs*feature_num+1):2*NMFs*feature_num),NMFs,feature_num);
b_mat=reshape(x((2*NMFs*feature_num+1):(2*NMFs*feature_num+feature_num*m))...
    ,feature_num,m);
c_mat=reshape(x((2*NMFs*feature_num+feature_num*m+1):(2*NMFs*feature_num+2*feature_num*m))...
    ,feature_num,m);
w_mat=reshape(x((2*NMFs*feature_num+2*feature_num*m+1):(2*NMFs*feature_num+3*feature_num*m))...
    ,feature_num,m);

%����ѵ���Ĳ����������Ԥ��
fwmodel_out = Fwnnmodel(Xtest,zeros(length(Xtest),1), NMFs,miu,sigma,b_mat,c_mat,w_mat);%��������Y��zeros(length(Xtest),1)
predict=fwmodel_out.Y_est;
yPredTest =InverseNormilzeData(predict,outputps);
end

%% Fwnnmodel
function  [fwmodel] = Fwnnmodel(S, Y, pre,miu,sigma,b_mat,c_mat,w_mat)
%%%Layout
%%%1.�����������ά�Ȳ����ڶ��㣨ģ��ǰ����
%%%2.����ģ��ǰ������������ģ������ȫ���У�
%%%3.�м䴦���������������
%%%4.���������������ɶ�Ӧ��С��������
%%%5.��ʧ����MSE
%%%6.����ѵ��������ƫ����
%%%7.�ݶ��㷨

%�������γ��
[sample_num,feature_num]=size(S);
%����������γ���Ƿ�ƥ��
[m_out,out_dim]=size(Y);
if(m_out~=sample_num)
    A=input("���ݲ�ƥ��");
end
if(out_dim~=1)
    B=input("�������һά����");
end
y_est=zeros(sample_num,1);
yita_all=zeros(sample_num,1);
yita=zeros(pre^feature_num,sample_num);
phai=zeros(pre^feature_num,sample_num);
index=zeros(pre^feature_num,feature_num);
for i=1:sample_num%ÿ�����ݰ����������ģ������
x=S(i,:);
%Layer2,����ģ��ǰ����,ÿһ�ж�Ӧһ��ǰ����
A_fuzz=Gauss_Func(x,miu,sigma);%A_fuzz(pre*feature_num)
%Layer3,����ģ������m��
m=pre^feature_num;
[fuzz_rule,index]=Fuzz_Rule(feature_num,pre,A_fuzz);%fuzz_rule=(pre^feature_num,1)
%Layer4,���ģ�������Ȩ��
yita_all(i)=sum(fuzz_rule(:));
yita(:,i)=fuzz_rule/yita_all(i);
%Layer5,��pre^feature_num����С��������
phai(:,i)=Wnn(x,feature_num,m,b_mat,c_mat,w_mat);%phai(m,1)
%Layer6,���
y_est(i)=(yita(:,i)')*phai(:,i);
end
fwmodel=struct('Y_est',y_est,'Y_true',Y,'X',S,'Miu_mat',miu, ...
    'Sigma_mat',sigma,'B_mat',b_mat,'C_mat',c_mat, ...
    'W_mat',w_mat,'Yita',yita,'Yita_all',yita_all,'Pre',pre, ...
    'Phai',phai,'Index',index);
end

%% Init_para
function [miu,sigma,b_mat,c_mat,w_mat]=Init_para(pre,feature_num,m)
%������1��ʼ����Ҳ�ɲ���c-clusteting
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
k=ones(m,feature_num);%��ʼ��index����
origin_k=ones(1,feature_num);
for ik=1:m%����index����
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
    %E��wƫ��(feature_num,m)
    pE_w=pE_w+pE_y_inside(i)*yita.*...
        ((one_mat-mid_w).*(exp(-1/2*mid_w)));
    %E��bƫ��(feature_num,m)
    mid_b=((x-fmodel.B_mat)./fmodel.C_mat).^2;
    pE_b=pE_b+pE_y_inside(i)*yita.*fmodel.W_mat...
        .*((x-fmodel.B_mat)./(fmodel.C_mat.^2)).*...
        (3*one_mat-mid_b).*exp(-1/2*mid_b);
    %E��cƫ��(feature_num,m)
    mid_c=((x-fmodel.B_mat).^2)./((fmodel.C_mat).^3);
    pE_c=pE_c+pE_y_inside(i)*yita.*fmodel.W_mat...
        .*mid_c.*(3*one_mat-mid_b).*exp(-1/2*mid_b);
    %y��Aƫ��(pre,feature_num)
    p_A_inside=Partial_a(fmodel.X(i,:),fmodel.Miu,...
        fmodel.Sigma,feature_num,pre,fmodel.Index,...
        fmodle.Phai(:,i),fmodel.Yita_all(i,1));
    %E��miuƫ��(pre,feature_num)
    mid_miu=((repmat(fmodel.X(i,:),pre,1)-fmodel.Miu)./...
        fmodel.Sigma).^2;
    pE_miu=pE_miu+pE_y_inside(i)*p_A_inside.*...
        (repmat(fmodel.X(i,:),pre,1)-fmodel.Miu)./...
        ((fmodel.Sigma.^2)).*exp(-1/2*mid_miu);
    %E��sigmaƫ��(pre,feature_num)
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
    %��y��A1,i1ƫ��
    p_A_inside=zeros(pre,feature_num);
    A_fuzz=Gauss_Func(x,miu,sigma);
for i=1:feature_num
    A_fuzz_d=A_fuzz;
    A_fuzz_d(:,i)=[];
    index(:,[1,i])=index(:,[i,1]);
    for j=1:pre
    %�Էǵ�һ��Ԫ��ȥ����Ӧ�У��ǵ�һ��Ԫ�ض�ӦindexֵΪj
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