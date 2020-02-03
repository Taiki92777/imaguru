#%%
# principle of majority voting
import numpy as np
np.argmax(np.bincount([0,0,1],weights=[0.2,0.2,0.6]))
# maximize the sum of probabilities
ex=np.array([[0.9,0.1],[0.8,0.2],[0.4,0.6]])
p=np.average(ex,axis=0,weights=[0.2,0.2,0.6])
print(p)
np.argmax(p)
#%%
# implementation
# load & label & split
from layers.classes import MajorityVoteClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

iris=datasets.load_iris()
x,y=iris.data[50:,[1,2]],iris.target[50:]
le=LabelEncoder()
y=le.fit_transform(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=1,stratify=y)
#%%
# Check abilities of LogisticRegression & DecisionTree & KNN
# using ROC AUC
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

clf1=LogisticRegression(penalty='l2',C=0.001,random_state=1)
clf2=DecisionTreeClassifier(max_depth=1,criterion='entropy',random_state=0)
clf3=KNeighborsClassifier(n_neighbors=1,p=2,metric='minkowski')

pipe1=Pipeline([['sc',StandardScaler()],['clf',clf1]])
pipe3=Pipeline([['sc',StandardScaler()],['clf',clf3]])
clf_labels=['Logistic regression','Decision tree','KNN']
print('10-fold cross validation:\n')
for clf, label in zip([pipe1,clf2,pipe3],clf_labels):
    scores=cross_val_score(estimator=clf,X=x_train,y=y_train,cv=10,scoring='roc_auc')
    print("ROC AUC: %0.2f(+/- %0.2f) [%s]"%(scores.mean(),scores.std(),label))

#%%
# Majority voting
mv_clf=MajorityVoteClassifier([pipe1,clf2,pipe3])
clf_labels+=['Majority voting']
all_clf=[pipe1,clf2,pipe3,mv_clf]
for clf, label in zip(all_clf,clf_labels):
    scores=cross_val_score(clf,X=x_train,y=y_train,cv=10,scoring='roc_auc')
    print("ROC AUC: %0.2f(+/- %0.2f) [%s]"%(scores.mean(),scores.std(),label))
