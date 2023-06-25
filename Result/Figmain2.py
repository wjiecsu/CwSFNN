import numpy as np
from   matplotlib import pyplot as plt  
import seaborn as sns
import pandas  as pd
Yall=pd.read_csv('./Yall.csv',encoding='gbk',header=None)
Yall=Yall.values
YReal=Yall[:,1]
YPred=Yall[:,0]
font={'family':'Times New Roman',
      'weight':'normal',
      'size':14}
colorset =['darkorange','red','dusty purple','greyish','green']
markerset=['^','v','o','d','s','*']
methodname=['CwSOFNN']
xIteration=[i for i in range(200)]

R2=[0.94]
RMSE=[0.014]

YpredTest=YPred
plt.figure(figsize=[9,2],dpi=300)
plt.plot(xIteration,YpredTest,color=sns.xkcd_rgb['red'],lw=3,label=methodname[i])
plt.plot(xIteration,YReal,color=sns.xkcd_rgb['blue'],lw=3,label='Real Vaule')
plt.grid(True)
plt.grid(True)
plt.xticks(fontproperties='Times New Roman',size=14)
plt.yticks(fontproperties='Times New Roman',size=14)
plt.legend(prop=font)
plt.xlabel(r'Samples',font)
plt.ylabel(r'Vaule',font)
bbox_props = dict(boxstyle="round",fc="w", ec="0.8",lw=1,alpha=0.9)
plt.text(100,0.53,r"$R^2$="+str(R2[i])+" RMSE="+str(RMSE[i]),
        fontsize=16,
        fontname='Times New Roman',
        color="k",
        verticalalignment ='top', 
        horizontalalignment ='center',
        bbox =bbox_props
    )
#plt.savefig('./AEPPrediction'+str(i)+'.png',bbox_inches = 'tight')



