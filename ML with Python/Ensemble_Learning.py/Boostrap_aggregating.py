#%%
# load & bagging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

df_wine=pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)
df_wine.columns=['Class label','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/315 of diluted wines','Proline']
df_wine=df_wine[df_wine['Class label']!=1]
y=df_wine['Class label'].values
x=df_wine[['Alcohol','OD280/315 of diluted wines']].values

le=LabelEncoder()
y=le.fit_transform(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1,stratify=y)

tree=DecisionTreeClassifier(criterion='entropy',max_depth=None,random_state=1)
bag=BaggingClassifier(base_estimator=tree,n_estimators=500,max_samples=1.0,max_features=1.0,bootstrap=True,bootstrap_features=False,n_jobs=1,random_state=1)
#%%
# compare abilities of DecisionTree & Bagging
from sklearn.metrics import accuracy_score
tree=tree.fit(x_train,y_train)
y_train_pred=tree.predict(x_train)
y_test_pred=tree.predict(x_test)
tree_train=accuracy_score(y_train,y_train_pred)
tree_test=accuracy_score(y_test,y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f'%(tree_train,tree_test))
bag=bag.fit(x_train,y_train)
y_train_pred=bag.predict(x_train)
y_test_pred=bag.predict(x_test)
bag_train=accuracy_score(y_train,y_train_pred)
bag_test=accuracy_score(y_test,y_test_pred)
print('Bagging train/test accuracies %.3f/%.3f'%(bag_train,bag_test))
#%%
# show decision plot
from layers.decisionregionplotfunction import plot_decision_regions_row_column
import matplotlib.pyplot as plt
plot_decision_regions_row_column(x_train,y_train,x1title='OD280/315 of diluted wines',x2title='Alcohol',classifiers=[tree,bag],clf_labels=['Decision tree','Bagging'],nrows=1,ncols=2,figsize=(8,3))
plt.show()
    