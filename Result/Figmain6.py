import numpy as np
from   matplotlib import pyplot as plt  
import seaborn as sns
import pandas  as pd
RMSE_TEP=pd.read_csv('./RMSE_TEP.csv',encoding='gbk',header=None)
Runtime_TEP=pd.read_csv('./Runtime_TEP.csv',encoding='gbk',header=None)
RMSE_TEP=RMSE_TEP.values
Runtime=Runtime_TEP.values
font={'family':'Times New Roman',
      'weight':'normal',
      'size':12}
colorset=['dark orange','gold','purple','black']
markerset=['^','v','o','d','s','*']

classname=['TSKFNN_EBP','MBGD_RDA','SOFNN_ALA','CwSOFNN']
subname=['(a)','(b)','(c)','(d)']
N=len(classname)
Index_number=np.array([1,2,3,4])
bar_width=0.9

XIndex=['TSKFNN_EBP','MBGD_RDA','SOFNN_ALA','CwSOFNN']
colorset=['dark orange','gold','purple','black','green']
plt.figure(figsize=[5,4],dpi=300)
for i in range(0,N,1):
    plt.bar(Index_number[i]+bar_width/2,height=Runtime[0,i],width=bar_width,color=sns.xkcd_rgb[colorset[i]],label=classname[i])
plt.grid(True)
plt.legend(prop=font)
plt.ylabel('Training Time(s)',font)
plt.yticks(fontproperties='Times New Roman',size=12)
plt.xticks(Index_number+bar_width/2,XIndex,fontproperties='Times New Roman',size=12)

plt.figure(figsize=[5,4],dpi=300)
for i in range(0,N,1):
    plt.bar(Index_number[i]+bar_width/2,height=RMSE_TEP[0,i]*1000,width=bar_width,color=sns.xkcd_rgb[colorset[i]],label=classname[i])
plt.grid(True)
plt.legend(prop=font)
plt.ylabel('RMSE( *1e-3)',font)
plt.ylim(12,25)
plt.yticks(fontproperties='Times New Roman',size=12)
plt.xticks(Index_number+bar_width/2,XIndex,fontproperties='Times New Roman',size=12)