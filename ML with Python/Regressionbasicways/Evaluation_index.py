#%%
# load
import pandas as pd

df=pd.read_csv('http://raw.githubusercontent.com/rasbt/python-machine-learning-book-2nd-edition/master/code/ch10/housing.data.txt',header=None,sep='\s+')
df.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
#%%
# split & linearregression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x=df.iloc[:,:-1].values
y=df['MEDV'].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
slr=LinearRegression()
slr.fit(x_train,y_train)
y_train_pred=slr.predict(x_train)
y_test_pred=slr.predict(x_test)
#%%
# residual plot
import matplotlib.pyplot as plt
plt.scatter(y_train_pred,y_train_pred-y_train,c='steelblue',marker='o',edgecolors='white',label='Training data')
plt.scatter(y_test_pred,y_test_pred-y_test,c='limegreen',marker='s',edgecolors='white',label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=-10,xmax=50,color='black',lw=2)
plt.xlim([-10,50])
plt.tight_layout()
plt.show()
#%%
# mean squared error
from sklearn.metrics import mean_squared_error
print('MSE train:%.3f, test:%.3f'%(mean_squared_error(y_train,y_train_pred),mean_squared_error(y_test,y_test_pred)))
#%%
# R**2
from sklearn.metrics import r2_score
print('R**2 train:%.3f,test:%.3f'%(r2_score(y_train,y_train_pred),r2_score(y_test,y_test_pred)))