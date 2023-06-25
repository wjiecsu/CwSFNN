import numpy as np
from   matplotlib import pyplot as plt  
import seaborn as sns
import pandas  as pd
Yall=pd.read_csv('./YPredTest_AEP.csv',encoding='gbk',header=None)
Yall=Yall.values
YReal=Yall[:,0]
YPred=Yall[:,[1,5,6,7]]
font={'family':'Times New Roman',
      'weight':'normal',
      'size':16}
colorset=['dark orange','gold','purple','black']
markerset=['^','v','o','d','s','*']
methodname=['TSKFNN_EBP','MBGD_RDA','SOFNN_ALA','CwSOFNN']
subname=['(a)','(b)','(c)','(d)']
xIteration=[i for i in range(85)]
markerIteration=[i for i in range(0,85,5)]
R2=[0.79, 0.93 ,0.87 ,0.94 ,0.86 ,0.94, 0.95]
RMSE=[0.77,0.45,0.61,0.42,0.64,0.41,0.37]
plt.figure(figsize=[18,6],dpi=300)
for i in range(0,4,1):
    ax=plt.subplot(2,2,i+1)
    plt.plot(xIteration,YReal,color=sns.xkcd_rgb['blue'],lw=3,label='Real Vaule')
    YpredTest=YPred[:,i]
    plt.plot(xIteration,YpredTest,color=sns.xkcd_rgb['red'],lw=3,label=methodname[i])   
    plt.plot(markerIteration,YpredTest[markerIteration],'^',markersize=6,markerfacecolor='white',color=sns.xkcd_rgb['red'],markeredgewidth=2)
    plt.plot(markerIteration,YReal[markerIteration],'o',markersize=6,markerfacecolor='white',color='blue',markeredgewidth=2)
    plt.grid(True)
    plt.grid(True)
    plt.xticks(fontproperties='Times New Roman',size=16)
    plt.yticks(fontproperties='Times New Roman',size=16)
    plt.legend(prop=font)
    plt.xlabel(r'Samples',font)
    plt.ylabel(r'Vaule',font)
    ax.set_xlabel(xlabel = subname[i],fontname='Times New Roman', fontsize=24)    
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.4)
#plt.savefig('./AEPPrediction'+str(i)+'.png',bbox_inches = 'tight')