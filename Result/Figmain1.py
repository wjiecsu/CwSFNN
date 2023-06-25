import numpy as np
from   matplotlib import pyplot as plt  
import seaborn as sns
import pandas  as pd
R2Test=pd.read_csv('./HyPerRuntime.csv',encoding='gbk',header=None)
R2Test=R2Test.values
plt.figure(figsize=[5,4],dpi=300)
font={'family':'Times New Roman',
      'weight':'normal',
      'size':14}
colorset =['blue','red','dusty purple','greyish','green']
markerset=['^','v','o','d','s','*']
PS=np.size(R2Test,1)
SS=np.size(R2Test,0)
methodname=[r'$\rho_\theta=$'+str(i) for i in [0.01,0.03,0.05,0.08,0.10]]
xIteration=[i*0.01+0.01 for i in range(0,SS)]
for i in range(0,PS,1):
    plt.plot(xIteration,R2Test[:,i],color=sns.xkcd_rgb[colorset[i]],lw=1,marker=markerset[i],markersize=5,label=methodname[i])
plt.xlabel(r'${S_\theta }$',font)
plt.ylabel(r'Time (s)',font)
plt.grid(True)
plt.grid(True)
plt.xticks(fontproperties='Times New Roman',size=14)
plt.yticks(fontproperties='Times New Roman',size=14)
plt.legend(prop=font)
