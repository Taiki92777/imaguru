#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df_wine=pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)
df_wine.columns=['Class label','Alchohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','OD280/OD315of diluted wines','Prolone']
#%%
x,y=df_wine.iloc[:,1:].values,df_wine.iloc[:,0].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0,stratify=y)
#%%
#標準化
from sklearn.preprocessing import StandardScaler

stdac=StandardScaler()
x_train_std=stdac.fit_transform(x_train)
x_test_std=stdac.transform(x_test)
#%%
#特徴量選択としてのL1正則化
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(penalty='l1',C=1.0)
lr.fit(x_train_std,y_train)
#%%
print('Training accuracy:',lr.score(x_train_std,y_train))
#%%
print('Test accuracy:',lr.score(x_test_std,y_test))
#%%
lr.intercept_
#%%
lr.coef_
#%%
import matplotlib.pyplot as plt

fig=plt.figure()
ax=plt.subplot(111)

colors=['blue','green','red','cyan','magenta','yellow','black','pink','lightgreen','lightblue','gray','indigo','orange']

weights,params=[],[]

for c in np.arange(-4.,6.):
    lr=LogisticRegression(penalty='l1',C=10.**c,random_state=0)
    lr.fit(x_train_std,y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights=np.array(weights)

for column, color in zip(range(weights.shape[1]),colors):
    plt.plot(params,weights[:,column],label=df_wine.columns[column+1],color=color)

plt.axhline(0,color='black',linestyle='--',linewidth=3)

plt.xlim([10**(-5),10**5])

plt.ylabel('weight coefficient')
plt.xlabel('C')

plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center',bbox_to_anchor=(1.38,1.03),ncol=1,fancybox=True)
plt.show()
#%%
#Sequential Backward Selection(逐次後退選択)による次元削減・特徴選択
#SBSの効果を示す（SBSクラスはlayers.classesにある）
from layers.classes import SBS
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=5)
sbs=SBS(knn,k_features=1)
sbs.fit(x_train_std,y_train) #SBS内部でもデータ分割されるが、元々のテストデータを分離しておくためにあらかじめ訓練データでSBSを適応しておく

#%%
#特徴量個数のリスト
k_feat=[len(k) for k in sbs.subsets_]
plt.plot(k_feat,sbs.scores_,marker='o')
plt.ylim([0.7,1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()
#%%
#抽出した特徴量のコラム表示
k3=list(sbs.subsets_[10])
print(df_wine.columns[1:][k3])
#%%
#13個全ての特徴量を用いたモデルの適合
knn.fit(x_train_std,y_train)
print('Training accuracy:',knn.score(x_train_std,y_train))
print('Test accuracy:',knn.score(x_test_std,y_test))
#%%
#抽出した3個の特徴量を用いたモデルの適合
knn.fit(x_train_std[:,k3],y_train)
print('Training accuracy:',knn.score(x_train_std[:,k3],y_train))
print('Test accuracy:',knn.score(x_test_std[:,k3],y_test))
