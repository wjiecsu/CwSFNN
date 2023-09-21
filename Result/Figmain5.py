import numpy as np
from   matplotlib import pyplot as plt  
import seaborn as sns
import pandas  as pd

ConfirmMatrix=pd.read_csv('./ConfirmMatrix.csv',encoding='gbk',header=None)
ConfirmMatrix=ConfirmMatrix.values
PersonSum=ConfirmMatrix[:,0]
RuleSet=np.insert(ConfirmMatrix[:,1],0,256)
RMSE=ConfirmMatrix[:,2]
font={'family':'Times New Roman',
      'weight':'normal',
      'size':10}
colorset=['dark orange','gold','purple','black']
markerset=['^','v','o','d','s','*']
subname=['(a)','(b)','(c)']
N=len(RMSE)
xIteration=[i for i in range(0,N,3)]
plt.figure(figsize=[15,4],dpi=300)
ax=plt.subplot(1,3,1)
plt.plot(xIteration,RMSE[xIteration],color=sns.xkcd_rgb['blue'],lw=3,label='RMSE')
plt.grid(True)
plt.grid(True)
plt.xticks(fontproperties='Times New Roman',size=10)
plt.yticks(fontproperties='Times New Roman',size=10)
plt.legend(prop=font)
plt.xlabel(r'Iterations',font)
plt.ylabel(r'RMSE',font)
ax=plt.subplot(1,3,2)
plt.plot(RuleSet,color=sns.xkcd_rgb['red'],lw=3,label='The number of Rules')
plt.grid(True)
plt.grid(True)
plt.xticks(fontproperties='Times New Roman',size=10)
plt.yticks(fontproperties='Times New Roman',size=10)
plt.legend(prop=font)
plt.xlabel(r'Iterations',font)
plt.ylabel(r'The number of Rules',font)
ax=plt.subplot(1,3,3)
plt.plot(PersonSum,color=sns.xkcd_rgb['black'],lw=3,label=r'${\rho _r}(e)$')
plt.grid(True)
plt.grid(True)
plt.xticks(fontproperties='Times New Roman',size=10)
plt.yticks(fontproperties='Times New Roman',size=10)
plt.legend(prop=font)
plt.xlabel(r'Iterations',font)
plt.ylabel(r'r'${\rho _r}(e)$'',font)
