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

df=pd.read_csv('http://archive.ics.uci.edu/ml/''machine-learning-databases/iris/iris.data',header=None)
df.tail()

y=df.iloc[0:100,4].values
y=np.where(y=='Iris-setosa',0,1)
x=df.iloc[0:100,[0,2]].values

lrgd=ccc.LogisticRegressionGD(eta=0.05,n_iter=1000,random_state=1)
lrgd.fit(x,y)

drp.plot_decision_regions(x=x,y=y,classifier=lrgd)

plt.xlabel('sepal length [standardaized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()