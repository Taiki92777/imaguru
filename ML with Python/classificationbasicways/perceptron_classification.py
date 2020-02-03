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


df=pd.read_csv('http://archive.ics.uci.edu/ml/''machine-learning-databases/iris/iris.data',header=None)
df.tail()

y=df.iloc[0:100,4].values
y=np.where(y=='Iris-setosa',-1,1)
x=df.iloc[0:100,[0,2]].values
'''
plt.scatter(x[:50,0],x[:50,1],color='red',marker='o',label='setosa')
plt.scatter(x[50:100,0],x[50:100,1],color='blue',marker='x',label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')

plt.legend(loc='upper left')

plt.show()

#グラフから線形でクラス分類できることがわかる
'''
ppn=ccc.Perceptron1(eta=0.1,n_iter=10)
ppn.fit(x,y)

plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='o')

plt.xlabel('Epochs')
plt.ylabel('Number of update')

plt.show()

print(ppn.errors_)