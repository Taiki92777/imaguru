#%%
# load & label & split
from layers.classes import MajorityVoteClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np


iris=datasets.load_iris()
x,y=iris.data[50:,[1,2]],iris.target[50:]
le=LabelEncoder()
y=le.fit_transform(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=1,stratify=y)
#%%
# make models ['Logistic regression','Decision tree','KNN','Majority voting']
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

clf1=LogisticRegression(penalty='l2',C=0.001,random_state=1)
clf2=DecisionTreeClassifier(max_depth=1,criterion='entropy',random_state=0)
clf3=KNeighborsClassifier(n_neighbors=1,p=2,metric='minkowski')

pipe1=Pipeline([['sc',StandardScaler()],['clf',clf1]])
pipe2=Pipeline([['sc',StandardScaler()],['clf',clf2]])
pipe3=Pipeline([['sc',StandardScaler()],['clf',clf3]])

mv_clf=MajorityVoteClassifier(classifiers=[pipe1,clf2,pipe3])

clf_labels=['Logistic regression','Decision tree','KNN','Majority voting']
all_clf=[pipe1,clf2,pipe3,mv_clf]
#%%
# Evaluate & Tuning
# ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt 
colors=['black','orange','blue','green']
linestyle=[':','--','-.','-']
fig,ax=plt.subplots()
fig.set_facecolor('white')
for clf,label,clr,ls in zip(all_clf,clf_labels,colors,linestyle):
    y_pred=clf.fit(x_train,y_train).predict_proba(x_test)[:,1]
    fpr,tpr,thresholds=roc_curve(y_true=y_test,y_score=y_pred)
    roc_auc=auc(x=fpr,y=tpr)
    ax.plot(fpr,tpr,color=clr,linestyle=ls,label='%s (auc = %0.2f)'%(label,roc_auc))

plt.legend(loc='lower right')
ax.plot([0,1],[0,1],linestyle='--',color='gray',linewidth=2)
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.grid(alpha=0.5)
plt.xlabel('False positive rate (FPR)')
plt.ylabel('True positive rate (TPR)')
plt.show()
#%%
# show decision-boundaries
from layers.decisionregionplotfunction import plot_decision_regions_row_column
plot_decision_regions_row_column(x_train,y_train,x1title='Sepal width [standardized]',x2title='Petal width [standardized]',classifiers=all_clf,clf_labels=clf_labels,nrows=2,ncols=2,figsize=(7,5))
plt.show()
#%%
# gain access to parameters
mv_clf.get_params()
# wow
#%%
# Tuning parameters of LogisticRegression__C & DecisionTree__depth
# gridsearch
from sklearn.model_selection import GridSearchCV
# params chosen from get_params()
params={'decisiontreeclassifier__max_depth':[1,2],'pipeline-1__clf__C':[0.001,0.1,100.0]}
grid=GridSearchCV(estimator=mv_clf,param_grid=params,cv=10,scoring='roc_auc')
grid.fit(x_train,y_train)

for r,_ in enumerate(grid.cv_results_['mean_test_score']):
    print("%0.3f +/- %0.2f %r"%(grid.cv_results_['mean_test_score'][r],grid.cv_results_['std_test_score'][r]/2,grid.cv_results_['params'][r]))

print('Best parameters: %s'% grid.best_params_)
print('Accuracy: %.2f'% grid.best_score_)