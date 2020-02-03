#%%
# load
import pandas as pd

df=pd.read_csv('http://raw.githubusercontent.com/rasbt/python-machine-learning-book-2nd-edition/master/code/ch10/housing.data.txt',header=None,sep='\s+')
df.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
df.head()
#%%
import numpy as np
import matplotlib.pyplot as plt
x=df[['RM']].values
y=df['MEDV'].values
#%%
# RANSAC  learn inlier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
ransac=RANSACRegressor(LinearRegression(),max_trials=100,min_samples=50,loss='absolute_loss',residual_threshold=5.0,random_state=0)
ransac.fit(x,y)
#%%
# plot
inlier_mask=ransac.inlier_mask_
outlier_mask=np.logical_not(inlier_mask)
line_x=np.arange(3,10,1)
line_y_ransac=ransac.predict(line_x[:,np.newaxis])
plt.scatter(x[inlier_mask],y[inlier_mask],c='steelblue',edgecolors='white',marker='o',label='Inliners')
plt.scatter(x[outlier_mask],y[outlier_mask],c='limegreen',edgecolors='white',marker='s',label='Outliners')
plt.plot(line_x,line_y_ransac,color='black',lw=2)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper left')
plt.show()
#%%
# slope & intercept
print('Slope: %.3f'% ransac.estimator_.coef_[0])
print('Intercept: %.3f'% ransac.estimator_.intercept_)