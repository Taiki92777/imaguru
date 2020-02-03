#%%
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
#%%
# load data & split & make pipeline
df=pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',header=None)
x,y=df.loc[:,2:].values,df.loc[:,1].values
le=LabelEncoder()
y=le.fit_transform(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,stratify=y,random_state=1)
pipe_lr=make_pipeline(StandardScaler(),PCA(n_components=2),LogisticRegression(random_state=1))
#%%
# stratified k-fold cross-validation
import numpy as np
from sklearn.model_selection import StratifiedKFold

kfold=StratifiedKFold(n_splits=10,random_state=1).split(x_train,y_train)
scores=[]
for k,(train,test) in enumerate(kfold):
    pipe_lr.fit(x_train[train],y_train[train])
    score=pipe_lr.score(x_train[test],y_train[test])
    scores.append(score)
    print('Fold: %2d, Class dist.: %s, Acc: %.3f' %(k+1, np.bincount(y_train[train]),score))

print('\nCV accuracy: %.3f +/- %.3f'%(np.mean(scores),np.std(scores)))
#%%
# calculate the accuracy of the model more simply
from sklearn.model_selection import cross_val_score
scores=cross_val_score(estimator=pipe_lr,X=x_train,y=y_train,cv=10,n_jobs=1)
print('CV accuracy scores: %s'% scores)
print('CV accuracy: %.3f +/- %.3f'% (np.mean(scores),np.std(scores)))