#%%
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
#%%
# load data & split & make pipeline
df=pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',header=None)
x,y=df.loc[:,2:].values,df.loc[:,1].values
le=LabelEncoder()
y=le.fit_transform(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,stratify=y,random_state=1)
#%%
# Grid search
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

pipe_svc=make_pipeline(StandardScaler(),SVC(random_state=1))
param_range=[0.0001,0.001,0.01,0.1,1,10,100,1000]
param_grid=[{'svc__C':param_range,'svc__kernel':['linear']},{'svc__C':param_range,'svc__gamma':param_range,'svc__kernel':['rbf']}]

gs=GridSearchCV(estimator=pipe_svc,param_grid=param_grid,scoring='accuracy',cv=10,n_jobs=-1)
gs=gs.fit(x_train,y_train)
print(gs.best_score_)
print('\n %s'% gs.best_params_)
#%%
# cross validation
clf=gs.best_estimator_
clf.fit(x_train,y_train) 
print('Test accuracy: %.3f' % clf.score(x_test,y_test))   

#%%
# nested cross-validation
from sklearn.model_selection import cross_val_score
import numpy as np
gs=GridSearchCV(estimator=pipe_svc,param_grid=param_grid,scoring='accuracy',cv=2)
scores=cross_val_score(gs,x_train,y_train,scoring='accuracy',cv=5)
print('CV accuracy: %.3f +/- %.3f'% (np.mean(scores),np.std(scores)))
#%%
# SVM vs Decision tree
from sklearn.tree import DecisionTreeClassifier
gs=GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),param_grid=[{'max_depth':[1,2,3,4,5,6,7,None]}],scoring='accuracy',cv=2)
scores=cross_val_score(gs,x_train,y_train,scoring='accuracy',cv=5)
print('CV accuracy: %.3f +/- %.3f'% (np.mean(scores),np.std(scores)))
