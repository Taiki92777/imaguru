#%%
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
#%%
df=pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',header=None)
x,y=df.loc[:,2:].values,df.loc[:,1].values
le=LabelEncoder()
y=le.fit_transform(y) # Benign-->0 Malignant-->1
le.classes_ 
#%%
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,stratify=y,random_state=1)
#%%
pipe_lr=make_pipeline(StandardScaler(),PCA(n_components=2),LogisticRegression(random_state=1))
pipe_lr.fit(x_train,y_train)
y_pred=pipe_lr.predict(x_test)
print('Test accuracy: %.3f' % pipe_lr.score(x_test,y_test))
