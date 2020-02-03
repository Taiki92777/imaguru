#%%
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
#%%
# load data & label them
df=pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',header=None)
x,y=df.loc[:,2:].values,df.loc[:,1].values
le=LabelEncoder()
y=le.fit_transform(y)
# Benign-->357  Malignant-->212
#%%
# make unbalanced group  Benign-->357 Malignant-->40
x_imb=np.vstack((x[y==0],x[y==1][:40]))
y_imb=np.hstack((y[y==0],y[y==1][:40]))
# just predict class 0
y_pred=np.zeros(y_imb.shape[0])
np.mean(y_pred==y_imb)*100
#%%
# upsample minority
from sklearn.utils import resample
print('Number of class 1 samples before:',x_imb[y_imb==1].shape[0])
# sampling with replacement
x_upsampled,y_upsampled=resample(x_imb[y_imb==1],y_imb[y_imb==1],replace=True,n_samples=x_imb[y_imb==0].shape[0],random_state=123)
print('Number of class 1 samples after:',x_upsampled.shape[0])
#%%
# make balanced group
x_bal=np.vstack((x[y==0],x_upsampled))
y_bal=np.hstack((y[y==0],y_upsampled))
y_pred=np.zeros(y_bal.shape[0])
np.mean(y_pred==y_bal)*100