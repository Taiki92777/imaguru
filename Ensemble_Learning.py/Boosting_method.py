#%%
# load & split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df_wine=pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)
df_wine.columns=['Class label','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/315 of diluted wines','Proline']
df_wine=df_wine[df_wine['Class label']!=1]
y=df_wine['Class label'].values
x=df_wine[['Alcohol','OD280/315 of diluted wines']].values

le=LabelEncoder()
y=le.fit_transform(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1,stratify=y)
#%%
# Adaboost implementation
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

tree=DecisionTreeClassifier(criterion='entropy',max_depth=1,random_state=1)
ada=AdaBoostClassifier(base_estimator=tree,n_estimators=500,learning_rate=0.1,random_state=1)
tree=tree.fit(x_train,y_train)
y_train_pred=tree.predict(x_train)
y_test_pred=tree.predict(x_test)
tree_train=accuracy_score(y_train,y_train_pred)
tree_test=accuracy_score(y_test,y_test_pred)
print('Decision tree train/test accuracies %.3f/%.3f'%(tree_train,tree_test))

ada=ada.fit(x_train,y_train)
y_train_pred=ada.predict(x_train)
y_test_pred=ada.predict(x_test)
ada_train=accuracy_score(y_train,y_train_pred)
ada_test=accuracy_score(y_test,y_test_pred)
print('AdaBoost train/test accuracies %.3f/%.3f'%(ada_train,ada_test))
#%%
# show decision regions
from layers.decisionregionplotfunction import plot_decision_regions_row_column
import matplotlib.pyplot as plt
plot_decision_regions_row_column(x_train,y_train,x1title='OD280/351 of diluted wined',x2title='Alcohol',classifiers=[tree,ada],clf_labels=['Decision tree','AdaBoost'],nrows=1,ncols=2,figsize=(8,3))
plt.show()