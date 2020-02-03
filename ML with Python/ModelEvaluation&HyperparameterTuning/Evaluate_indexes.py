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
#%%
# find best estimator & apply to the training data
clf=gs.best_estimator_
clf.fit(x_train,y_train)
#%%
# other scores
'''
   P  N  
P  TP FN
N  FP TN
'''
from sklearn.metrics import confusion_matrix
y_pred=clf.predict(x_test)
confmat=confusion_matrix(y_true=y_test,y_pred=y_pred)
print(confmat) 
#%%
# plot them
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(2.5,2.5))
fig.set_facecolor('white')
ax.matshow(confmat,cmap=plt.cm.Blues,alpha=0.3) # make heatmap
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j,y=i,s=confmat[i,j],va='center',ha='center')

plt.xlabel('predicted label')
plt.ylabel('true label')
plt.tight_layout
plt.show()
#%%
# ERR ACC TPR FPR PRE REC F1-score
from sklearn.metrics import precision_score, recall_score, f1_score
print('Precision: %.3f'% precision_score(y_true=y_test,y_pred=y_pred))
print('Recall: %.3f'% recall_score(y_true=y_test,y_pred=y_pred))
print('F1: %.3f'% f1_score(y_true=y_test,y_pred=y_pred))
#%%
# try other scores
from sklearn.metrics import make_scorer
scorer=make_scorer(f1_score,pos_label=0)
gs=GridSearchCV(estimator=pipe_svc,param_grid=param_grid,scoring=scorer,cv=10,n_jobs=-1)
gs=gs.fit(x_train,y_train)
print(gs.best_score_)
print('\n %s' %gs.best_params_)
#%%
# ROC curve
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve,auc
from scipy import interp
import numpy as np
pipe_lr=make_pipeline(StandardScaler(),PCA(n_components=2),LogisticRegression(penalty='l2',random_state=1,C=100.0))
cv=list(StratifiedKFold(n_splits=3,random_state=1).split(x_train,y_train))
fig,ax=plt.subplots(figsize=(7,5))
fig.set_facecolor('white')
mean_tpr=0.0
mean_fpr=np.linspace(0,1,100)

for i,(train,test)in enumerate(cv):
    probas=pipe_lr.fit(x_train[train],y_train[train]).predict_proba(x_train[test])
    fpr,tpr,thresholds=roc_curve(y_train[test],probas[:,1],pos_label=1)
    mean_tpr+=interp(mean_fpr,fpr,tpr)
    mean_tpr[0]=0.0
    roc_auc=auc(fpr,tpr)
    ax.plot(fpr,tpr,label='ROC fold %d (area= %0.2f)'%(i+1,roc_auc))

ax.plot([0,1],[0,1],linestyle='--',color=(0.6,0.6,0.6),label='random guessing')
mean_tpr/=len(cv)
mean_tpr[-1]=1.0
mean_auc=auc(mean_fpr,mean_tpr)
ax.plot(mean_fpr,mean_tpr,'k--',label='mean ROC (area = %0.2f)'%mean_auc,lw=2)
ax.plot([0,0,1],[0,1,1],linestyle=':',color='black',label='perfect performance')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend(loc='lower right')
plt.tight_layout
plt.show()
#%%
# index for multi-classification
# micro-means macro-means
pre_scorer=make_scorer(score_func=precision_score,pos_label=1,greater_is_better=True,average='micro')
# default-->macro   