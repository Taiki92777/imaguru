#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
'''
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
'''
import layers
from layers import classes as ccc
from layers import decisionregionplotfunction as drp

#%%
df=pd.read_csv('http://archive.ics.uci.edu/ml/''machine-learning-databases/iris/iris.data',header=None)
df.tail()

y=df.iloc[0:100,4].values
y=np.where(y=='Iris-setosa',-1,1)
x=df.iloc[0:100,[0,2]].values
"""
#↓は正規化なしのADALINE
fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(10,4))
ada1=ccc.AdalineGD(eta=0.01,n_iter=10).fit(x,y)
ax[0].plot(range(1,len(ada1.cost_)+1),np.log10(ada1.cost_),marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(sumsquarederror)')
ax[0].set_title('adaline-eta 0.01')

ada2=ccc.AdalineGD(eta=0.0001,n_iter=10).fit(x,y)
ax[1].plot(range(1,len(ada2.cost_)+1),ada2.cost_,marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('sumsquarederror')
ax[1].set_title('adaline-eta 0.0001')

plt.show()
"""

#%%
#↓は正規化済み＋決定領域を示すグラフが出てくる（決定領域のグラフの仕組みはまだわかってない）
x_std=np.copy(x)
x_std[:,0]=(x[:,0]-x[:,0].mean())/x[:,0].std()
x_std[:,1]=(x[:,1]-x[:,1].mean())/x[:,1].std()
#%%
#ada=ccc.AdalineGD(n_iter=15,eta=0.01)
#ada.fit(x_std,y)
ads=ccc.AdalineGD_stochasticgradientdescent(n_iter=15,eta=0.01)
ads.fit(x_std,y)
#%%
#drp.plot_decision_regions(x_std,y,classifier=ada)
drp.plot_decision_regions(x_std,y,classifier=ads)

plt.xlabel('sepal length[standardized]')
plt.ylabel('petal length[standardized]')

plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
#%%
plt.plot(range(1,len(ads.cost_)+1),ads.cost_,marker='o')

plt.xlabel('Epochs')
plt.ylabel('sumsquarederror')

plt.tight_layout()
plt.show()