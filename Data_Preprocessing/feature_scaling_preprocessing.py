#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df_wine=pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)
df_wine.columns=['Class label','Alchohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315of diluted wines','Prolone']
#%%
x,y=df_wine.iloc[:,1:].values,df_wine.iloc[:,0].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0,stratify=y)
#%%
#正規化
from sklearn.preprocessing import MinMaxScaler

mms=MinMaxScaler()
x_train_norm=mms.fit_transform(x_train)
x_test_norm=mms.transform(x_test)
#%%
#標準化
from sklearn.preprocessing import StandardScaler

stdac=StandardScaler()
x_train_std=stdac.fit_transform(x_train)
x_test_std=stdac.transform(x_test)