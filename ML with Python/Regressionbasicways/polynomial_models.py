#%%
# 2D 3D
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

df=pd.read_csv('http://raw.githubusercontent.com/rasbt/python-machine-learning-book-2nd-edition/master/code/ch10/housing.data.txt',header=None,sep='\s+')
df.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
x=df[['LSTAT']].values
y=df['MEDV'].values
regr=LinearRegression()

quadratic=PolynomialFeatures(degree=2)
cubic=PolynomialFeatures(degree=3)
x_quad=quadratic.fit_transform(x)
x_cubic=cubic.fit_transform(x)

x_fit=np.arange(x.min(),x.max(),1)[:,np.newaxis]
regr=regr.fit(x,y)
y_lin_fit=regr.predict(x_fit)
linear_r2=r2_score(y,regr.predict(x))

regr=regr.fit(x_quad,y)
y_quad_fit=regr.predict(quadratic.fit_transform(x_fit))
quad_r2=r2_score(y,regr.predict(x_quad))

regr=regr.fit(x_cubic,y)
y_cubic_fit=regr.predict(cubic.fit_transform(x_fit))
cubic_r2=r2_score(y,regr.predict(x_cubic))


plt.scatter(x,y,label='training points',color='lightgray')
plt.plot(x_fit,y_lin_fit,label='linear (d=1), $R^2=%.2f$'%linear_r2,color='blue',lw=2,linestyle=':')
plt.plot(x_fit,y_quad_fit,label='quadratic (d=2), $R^2=%.2f$'%quad_r2,color='red',lw=2,linestyle='-')
plt.plot(x_fit,y_cubic_fit,label='cubic (d=3), $R^2=%.2f$'%cubic_r2,color='green',lw=2,linestyle='--')
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper right')
plt.show()
#%%
# 多項式がいいとは限らない。試しに対数関数で見てみると
x_log=np.log(x)
y_sqrt=np.sqrt(y)

x_fit=np.arange(x_log.min()-1,x_log.max()+1,1)[:,np.newaxis]
regr=regr.fit(x_log,y_sqrt)
y_lin_fit=regr.predict(x_fit)
linear_r2=r2_score(y_sqrt,regr.predict(x_log))

plt.scatter(x_log,y_sqrt,label='training points',color='lightgray')
plt.plot(x_fit,y_lin_fit,label='linear (d=1), $R^2=%.2f$'%linear_r2,color='blue',lw=2)
plt.xlabel('log(% lower status of the population [LSTAT])')
plt.ylabel('$\sqrt{Price \; in \; \$1000s [MEDV]}$')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
#%%
# Decision tree regression
from sklearn.tree import DecisionTreeRegressor
x=df[['LSTAT']].values
y=df['MEDV'].values
tree=DecisionTreeRegressor(max_depth=3)
tree.fit(x,y)
sort_idx=x.flatten().argsort()

from layers.decisionregionplotfunction import lin_regplot

lin_regplot(x[sort_idx],y[sort_idx],tree)
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.show()
#%%
# Random tree regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

x=df.iloc[:,:-1].values
y=df['MEDV'].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=1)

forest=RandomForestRegressor(n_estimators=1000,criterion='mse',random_state=1,n_jobs=-1)
forest.fit(x_train,y_train)
y_train_pred=forest.predict(x_train)
y_test_pred=forest.predict(x_test)

print('MSE train: %.3f, test: %.3f'%(mean_squared_error(y_train,y_train_pred),mean_squared_error(y_test,y_test_pred)))
print('R2 train: %.3f, test: %.3f'%(r2_score(y_train,y_train_pred),r2_score(y_test,y_test_pred)))
#%%
# residual plot
plt.scatter(y_train_pred,y_train_pred-y_train,c='steelblue',marker='o',edgecolors='white',s=35,alpha=0.9,label='Training data')
plt.scatter(y_test_pred,y_test_pred-y_test,c='limegreen',marker='s',edgecolors='white',s=35,alpha=0.9,label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=-10,xmax=50,color='black',lw=2)
plt.xlim([-10,50])
plt.tight_layout()
plt.show()
