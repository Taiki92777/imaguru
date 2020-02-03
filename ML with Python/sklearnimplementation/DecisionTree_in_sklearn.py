#%%
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import layers
from layers import decisionregionplotfunction as drp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data

#%%
iris=datasets.load_iris()
x=iris.data[:,[2,3]]
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1,stratify=y)
x_combined=np.vstack((x_train,x_test))
y_combined=np.hstack((y_train,y_test))

#%%
tree=DecisionTreeClassifier(criterion='gini',max_depth=4,random_state=1)
tree.fit(x_train,y_train)

#%%
drp.plot_decision_regions(x_combined,y_combined,classifier=tree,test_idx=range(105,150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

#%%
dot_data=export_graphviz(tree,filled=True,rounded=True,class_names=['Setosa','Versicolor','Virginica'],feature_names=['petal length','petal width'],out_file=None)
graph=graph_from_dot_data(dot_data)
graph.write_png('tree.png')