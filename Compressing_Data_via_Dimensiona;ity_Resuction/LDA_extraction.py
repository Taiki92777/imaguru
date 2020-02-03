#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
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
#mean vecs
np.set_printoptions(precision=4)
mean_vecs=[]
for label in range(1,4):
    mean_vecs.append(np.mean(x_train_std[y_train==label],axis=0))
    print('MV %s: %s\n' % (label,mean_vecs[label-1]))
#%%
# within-class scatter matrix
d=13
S_W=np.zeros((d,d))
for label, mv in zip(range(1,4), mean_vecs):
    class_scatter=np.zeros((d,d))
    for row in x_train_std[y_train==label]:
        row,mv=row.reshape(d,1),mv.reshape(d,1)
        class_scatter+=(row-mv).dot((row-mv).T)
    S_W+=class_scatter

print('Within-class scatter matrix: %sX%s' % (S_W.shape[0],S_W.shape[1]))
#%%
# check class label distribution
print('Class label distribution: %s' % np.bincount(y_train)[1:])
#%%
# scaled Sw(scaling===>>>covariance)
d=13
S_W=np.zeros((d,d))
for label, mv in zip(range(1,4), mean_vecs):
    class_scatter=np.cov(x_train_std[y_train==label].T)
    S_W+=class_scatter

print('Scaled-Within-class scatter matrix: %sX%s' % (S_W.shape[0],S_W.shape[1]))
#%%
# Between-Class scatter matrix
mean_overall=np.mean(x_train_std,axis=0)
d=13
S_B=np.zeros((d,d))
for i,mean_vec in enumerate(mean_vecs):
    n=x_train[y_train==i+1,:].shape[0] #class=i+1のサンプル数
    mean_vec=mean_vec.reshape(d,1)
    mean_overall=mean_overall.reshape(d,1)
    S_B+=n*(mean_vec-mean_overall).dot((mean_vec-mean_overall).T)

print('Between-class scatter matrix: %sX%s' % (S_B.shape[0],S_B.shape[1]))
#%%
# calculate eigen factors and sort eigen values
eigen_vals,eigen_vecs=np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
eigen_pairs=[(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs=sorted(eigen_pairs,key=lambda k:k[0],reverse=True)
print('Eigenvalues in descending order:\n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])
# Sbはランクが1以下の行列をクラス数cだけ足したものなので、最大c-1しか固有値は存在しない
#%%
# dsicriminability
tot=sum(eigen_vals.real)
discr=[(i/tot) for i in sorted(eigen_vals.real,reverse=True)]
cum_discr=np.cumsum(discr)
#%%
# show figure(discriminability & cumlative dicriminability)
plt.bar(range(1,14),discr,alpha=0.5,align='center',label='discriminability')
plt.step(range(1,14),cum_discr,where='mid',label='cumulative discriminability') 
plt.ylabel('discriminability ratio') 
plt.xlabel('Linear discriminants')
plt.ylim([-0.1,1.1])
plt.legend(loc='best')
plt.tight_layout()
plt.show() 
#%%
# select most important eigen factors
w=np.hstack((eigen_pairs[0][1][:,np.newaxis].real,eigen_pairs[1][1][:,np.newaxis].real))
print('Matrix W:\n',w)
#%%
# X'=XW
x_train_lda=x_train_std.dot(w)
colors=['r','b','g']
markers=['s','x','o']
for l,c,m in zip(np.unique(y_train),colors,markers):
    plt.scatter(x_train_lda[y_train==l,0],x_train_lda[y_train==l,1],c=c,label=l,marker=m)

plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout
plt.show()
#%%
# Sklearn implementation
lda=LDA(n_components=2)
x_train_lda_sk=lda.fit_transform(x_train_std,y_train)
lr=LogisticRegression()
lr=lr.fit(x_train_lda_sk,y_train)
#%%
# show figure
pdr(x_train_lda_sk,y_train,classifier=lr)
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()
#%%
# test check
x_test_lda_sk=lda.transform(x_test_std)
pdr(x_test_lda_sk,y_test,classifier=lr)
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend(loc='lower right')
plt.tight_layout
plt.show()
