#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df_wine=pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)
df_wine.columns=['Class label','Alchohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315of diluted wines','Prolone']
#%%
x,y=df_wine.iloc[:,1:].values,df_wine.iloc[:,0].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0,stratify=y)
#%%
feat_labels=df_wine.columns[1:]
forest=RandomForestClassifier(n_estimators=500,random_state=1)
forest.fit(x_train,y_train)
#%%
#特徴量の重要度を抽出
importances=forest.feature_importances_
#重要度の降順で特徴量のインデックスを抽出
indices=np.argsort(importances)[::-1]
for f in range(x_train.shape[1]):
    print("%2d) %-*s %f" %(f+1,30,feat_labels[indices[f]],importances[indices[f]]))

#%%
#図示
plt.title('Feature Importances')
plt.bar(range(x_train.shape[1]),importances[indices],align='center')
plt.xticks(range(x_train.shape[1]),feat_labels[indices],rotation=90)
plt.xlim([-1,x_train.shape[1]])
plt.show()
#%%
#SelectFromModelで特徴選択
from sklearn.feature_selection import SelectFromModel
sfm=SelectFromModel(forest,threshold=0.1,prefit=True)
x_selected=sfm.transform(x_train)
print('Number of samples that meet this criterion:',x_selected.shape[0])

for f in range(x_selected.shape[1]):
    print("%2d) %-*s %f" % (f+1,30,feat_labels[indices[f]],importances[indices[f]]))
