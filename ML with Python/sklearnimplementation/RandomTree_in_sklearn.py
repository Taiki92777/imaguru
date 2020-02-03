#%%
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import layers
from layers import decisionregionplotfunction as drp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier

#%%
iris=datasets.load_iris()
x=iris.data[:,[2,3]]
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1,stratify=y)
x_combined=np.vstack((x_train,x_test))
y_combined=np.hstack((y_train,y_test))

#%%
forest=RandomForestClassifier(criterion='gini',n_estimators=25,random_state=1,n_jobs=2)
forest.fit(x_train,y_train)

#%%
drp.plot_decision_regions(x_combined,y_combined,classifier=forest,test_idx=range(105,150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
