#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from layers.decisionregionplotfunction import plot_decision_regions as pdr
#%%
# data loading
df_wine=pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)
x,y=df_wine.iloc[:,1:].values,df_wine.iloc[:,0].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,stratify=y,random_state=0)
#%%
# standard scaling
sc=StandardScaler()
x_train_std=sc.fit_transform(x_train)
x_test_std=sc.transform(x_test)
#%%
# make covariance matrix
cov_mat=np.cov(x_train_std.T)
eigen_vals,eigen_vecs=np.linalg.eig(cov_mat)
print('\nEigenvalues \n%s' % eigen_vals)
#%%
# variance explained ratio(to show importances) 
tot=sum(eigen_vals)
var_exp=[(i/tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp=np.cumsum(var_exp)
#%%
# show figure(variance explained ratio & cumlative variance explained ratio)
plt.bar(range(1,14),var_exp,alpha=0.5,align='center',label='individual explained variance')
plt.step(range(1,14),cum_var_exp,where='mid',label='cumulative explained variance') 
plt.ylabel('Explaned variance ratio') 
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show() 
#%%
# make (eigen value,eigen vectol)tuple list & sort it
eigen_pairs=[(np.abs(eigen_vals[i]),eigen_vecs[:,i])for i in range(len(eigen_vals))]
eigen_pairs.sort(key=lambda k: k[0],reverse=True)
#%%
# extract two eigen vectols which eigen values are biggest
w=np.hstack((eigen_pairs[0][1][:,np.newaxis],eigen_pairs[1][1][:,np.newaxis]))
print('Matrix W:\n',w)
#%%
# 13 dimensions-->2 dimensions
x_train_pca=x_train_std.dot(w)
#%%
# show figure
colors=['r','b','g']
markers=['s','x','o']
for l,c,m in zip(np.unique(y_train),colors,markers):
    plt.scatter(x_train_pca[y_train==l,0],x_train_pca[y_train==l,1],c=c,label=l,marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout
plt.show()
#%%
# sklearn implementation
pca=PCA(n_components=2)
lr=LogisticRegression()
x_train_pca_sk=pca.fit_transform(x_train_std)
x_test_pca_sk=pca.transform(x_test_std)
lr.fit(x_train_pca_sk,y_train)
#%%
# show figure
pdr(x_train_pca_sk,y_train,classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
#%%
# fit test-->plot
pdr(x_test_pca_sk,y_test,classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
#%%
# explained variance ratio
pca=PCA(n_components=None)
x_train_pca_sk=pca.fit_transform(x_train_std)
pca.explained_variance_ratio_
