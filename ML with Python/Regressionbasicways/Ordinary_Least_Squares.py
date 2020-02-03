#%%
# load
import pandas as pd

df=pd.read_csv('http://raw.githubusercontent.com/rasbt/python-machine-learning-book-2nd-edition/master/code/ch10/housing.data.txt',header=None,sep='\s+')
df.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
df.head()
#%%
# LinearRegression
import numpy as np
from sklearn.preprocessing import StandardScaler
from layers.classes import LinearRegressionGD
import matplotlib.pyplot as plt
x=df['RM'].values
y=df['MEDV'].values
sc_x=StandardScaler()
sc_y=StandardScaler()
x_std=sc_x.fit_transform(x[:,np.newaxis]) #x=df[['RM']].values でも同じこと
y_std=sc_y.fit_transform(y[:,np.newaxis]).flatten()
lr=LinearRegressionGD()
lr.fit(x_std,y_std)

plt.plot(range(1,lr.n_iter+1),lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()
#%%
# show hows everything going
from layers.decisionregionplotfunction import lin_regplot
lin_regplot(x_std,y_std,lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000s [MEDV] (standardized)')
plt.show()
#%%
# original scale
num_rooms_std=sc_x.transform([[5.0]]) # expected 2D array
price_std=lr.predict(num_rooms_std)
print("Price in $1000s: %.3f"%sc_y.inverse_transform(price_std))
#%%
# 標準化されていればバイアス(w[0])は常に０である。なので本当は更新しなくてもいい
print('Slope: %.3f'%lr.w_[1])
print('Intercept: %.3f'%lr.w_[0])
#%%
# sklearn implementation
from sklearn.linear_model import LinearRegression
slr=LinearRegression()
x=x[:,np.newaxis]
slr.fit(x,y)
y_pred=slr.predict(x)
print('Slope: %.3f'%slr.coef_[0])
print('Intercept: %.3f'%slr.intercept_)
lin_regplot(x,y,slr)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')
plt.show()