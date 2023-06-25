import numpy as np
from   matplotlib import pyplot as plt  
import seaborn as sns
import pandas  as pd
from matplotlib.pyplot import MultipleLocator
plt.rc('font', family='Times New Roman')
plt.rcParams['font.size']=8  # 设置字体大小
font={'family':'Times New Roman',
      'weight':'normal',
      'size':10}


AEPWeightMatrix=pd.read_csv('./AEPWeightMatrix.csv',encoding='gbk',header=None)
Yall=pd.read_csv('./AEPYall.csv',encoding='gbk',header=None)
RMSE=pd.read_csv('./AEPRMSE.csv',encoding='gbk',header=None)
Peacorr=pd.read_csv('./AEPpearsoncorr.csv',encoding='gbk',header=None)
Peasum=pd.read_csv('./AEPpearsonsum.csv',encoding='gbk',header=None)
AEPWeightMatrix=AEPWeightMatrix.values
Yall=Yall.values
RMSE=RMSE.values
Peacorr=Peacorr.values
Peasum =Peasum.values
YReal=Yall[:,0]
YPred=Yall[:,1]


# 第一个图
plt.figure(figsize=[5,4],dpi=300)
ax=plt.gca()
h=sns.heatmap(AEPWeightMatrix, cmap='Reds', annot=False,xticklabels=False,yticklabels=False)
plt.tick_params(axis='both',which='major',labelsize=10)
plt.xlabel(r'Rules',font)
plt.ylabel(r'Samples',font)
plt.xticks([1,20,40,60,80,100],[1,20,40,60,80,100],fontproperties='Times New Roman',size=8)
plt.yticks([1,20,40,60,80],[1,20,40,60,80],fontproperties='Times New Roman',size=8)


#第三个图
plt.figure(figsize=[9,3],dpi=300)
xIteration=[i for i in range(85)]
YpredTest=YPred
plt.plot(xIteration,YpredTest,color=sns.xkcd_rgb['red'],lw=1.5,label='WCA-NARX')
plt.plot(xIteration,YReal,color=sns.xkcd_rgb['blue'],lw=1.5,label='Real Vaule')
plt.grid(True)
plt.xticks(fontproperties='Times New Roman',size=14)
plt.yticks(fontproperties='Times New Roman',size=14)
plt.legend(prop=font)
plt.xlabel(r'Samples',font)
plt.ylabel(r'Vaule',font)
bbox_props = dict(boxstyle="round",fc="w", ec="0.8",lw=1,alpha=0.9)
plt.text(40,2.5,r"$R^2$="+str(0.95)+" RMSE="+str(0.37),
        fontsize=16,
        fontname='Times New Roman',
        color="k",
        verticalalignment ='top', 
        horizontalalignment ='center',
        bbox =bbox_props
    )

#第二个图
font={'family':'Times New Roman',
      'weight':'normal',
      'size':14}
plt.figure(figsize=[5,4],dpi=300)
xIteration1=[i for i in range(100)]
fig,ax1=plt.subplots()
plt.xticks(fontproperties='Times New Roman',size=14)
line1,=ax1.plot(xIteration1,RMSE,color=sns.xkcd_rgb['blue'],lw=1.5,label='Training RMSE')
ax1.grid(True)
ax1.set_xlabel('Epochs',font)
ax1.set_ylabel('RMSE value',font)
plt.yticks(fontproperties='Times New Roman',size=14)
ax2=ax1.twinx()
line2,=ax2.plot(xIteration1,Peasum.squeeze(),color=sns.xkcd_rgb['red'],lw=1.5,label='Training Pearson correlation ')
ax2.grid(True)
ax2.set_ylabel('Sum of Pearson correlation value',font)
plt.yticks(fontproperties='Times New Roman',size=14)
plt.legend(handles=[line1,line2],prop=font,loc=[0.35,0.5])