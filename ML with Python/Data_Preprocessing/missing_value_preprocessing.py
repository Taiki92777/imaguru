#%%
import pandas as pd
from io import StringIO
#欠損値特定
csv_data='''A,B,C,D
            1.0,2.0,3.0,4.0
            5.0,6.0,,8.0
            10.0,11.0,12.0,'''
df=pd.read_csv(StringIO(csv_data))
df
#%%
df.isnull().sum()
#%%
#欠損値除去
df.dropna()
#%%
df.dropna(axis=1)
#%%
df.dropna(how='all')
#%%
df.dropna(thresh=4)
#%%
df.dropna(subset=['C'])
#%%
#欠損値補完
from sklearn.preprocessing import Imputer

imr=Imputer(missing_values='NaN',strategy='mean',axis=0)
imr=imr.fit(df.values)
imputed_data=imr.transform(df.values)
imputed_data