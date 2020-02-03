#%%
import pandas as pd

df=pd.read_csv('http://raw.githubusercontent.com/rasbt/python-machine-learning-book-2nd-edition/master/code/ch10/housing.data.txt',header=None,sep='\s+')
df.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
df.head()
#%%
# 'MEDV'(住宅価格)についてこれから回帰予測したい
# Exploratory Data Analysis
import matplotlib.pyplot as plt
import seaborn as sns

cols=['LSTAT','INDUS','NOX','RM','MEDV']
sns.pairplot(df[cols],size=2.5).savefig('housing_pairplot.png')
plt.tight_layout()
plt.show()
#%%
# Pearson product-moment correlation coefficient
import numpy as np
cm=np.corrcoef(df[cols].values.T)
print(cm)
sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':15},yticklabels=cols,xticklabels=cols)
plt.tight_layout()
plt.savefig('housing_corrcoeff.png')
plt.show()