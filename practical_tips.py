#%%
# gini entropy error
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x=np.array([1,2,3])
print(x.reshape(1,-1))

def gini(p):
    return p*(1-p)+(1-p)*(1-(1-p))

def entropy(p):
    return -p*np.log2(p)-(1-p)*np.log2((1-p))

def error(p):
    return 1-np.max([p,1-p])

x=np.arange(0.0,1.0,0.01)

ent=[entropy(p) if p!=0 else None for p in x]
sc_ent=[e*0.5 if e else None for e in ent]
err=[error(i) for i in x]

fig=plt.figure()
ax=plt.subplot(111)

for i,lab,ls,c, in zip([ent,sc_ent,gini(x),err],['Entropy','Entropy(scaled)','Gini Impurity','Misclassification Error'],['-','-','--','-.'],['black','lightgray','red','green']):
    line=ax.plot(x,i,label=lab,linestyle=ls,lw=2,color=c)

ax.legend(loc='upper center',bbox_to_anchor=(0.5,1.15),ncol=5,fancybox=True,shadow=False)

ax.axhline(y=0.5,linewidth=1,color='k',linestyle='--')
ax.axhline(y=1.0,linewidth=1,color='k',linestyle='--')

plt.ylim([0,1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
plt.show()
#%%
# meshgrid ravel c_ array+T
import numpy as np
x1=np.array([1,2,3,4,5])
x2=np.array([1,2,3,4,5])
xx1,xx2=np.meshgrid(x1,x2)
print(xx1)
print(xx2)
print(xx1.ravel())
print(xx2.ravel())
print(np.c_[xx1.ravel(),xx2.ravel()])
print(np.array([xx1.ravel(),xx2.ravel()]))
print(np.array([xx1.ravel(),xx2.ravel()]).T)
#%%
import numpy as np
from itertools import product

nrows=1
ncols=2
a=range(nrows)
b=range(ncols)
print(a)
print(b)
for idx in product(a,b):
    print([idx[0],idx[1]])
#%%
import matplotlib.pyplot as plt
import numpy as np
x=np.arange(0,10,0.1)
y=np.power(x,2)
fig,ax=plt.subplots(nrows=2,ncols=2,figsize=(10,10))
fig.set_facecolor('white')
ax[0,0].scatter(x,y,c='blue',marker='^',s=50)
ax[0,1].scatter(x,y,c='green',marker='^',s=50)
ax[1,0].scatter(x,y,c='red',marker='^',s=50)
ax[1,1].scatter(x,y,c='yellow',marker='^',s=50)
plt.text(10,10,s='chee',ha='center',va='center',fontsize=12)
plt.show()
#%%
import sys
import pprint
pprint.pprint(sys.path)
#%%
import sys
is64Bit=sys.maxsize>2**32
if is64Bit:
    print('64bit')
else:
    print('32bit')
#%%
import numpy as np
i=np.array([1,2,3,4])
print(i**2)
#%%
import numpy as np
x=np.random.RandomState(seed=0).randn(2,2)
print(x)
#%%
import numpy as np
x=np.random.randn(25,300,6,4)
print((25,75,4)+(6,4))
print((len(x), x.shape[1] // 4, 4) )
print((len(x), x.shape[1] // 4, 4) +x.shape[2:])
r = x.reshape((len(x), x.shape[1] // 4, 4) + x.shape[2:])
print(r.shape)
print([r[:, :, i] for i in range(4)])
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1=pd.read_csv('historical_data_2014.csv')
df2=pd.read_csv('historical_data_2015.csv')
df3=pd.read_csv('historical_data_2016.csv')
df4=pd.read_csv('historical_data_2017.csv')
df5=pd.read_csv('historical_data_2018.csv')
df6=pd.read_csv('historical_data_2019.csv')

df=pd.concat([df1,df2,df3,df4,df5,df6])

df.head(5)

df=df.drop(['PT_TOKYO','GOLD_NY','PT_NY','USDJPY'],axis=1)

values=df.iloc[:,1].values
plt.plot(values)

# 階差取得
def _diff(values):
    diff=values[1:]-values[:-1]
    return diff
diff=_diff(values)
plt.plot(diff)

print(len(diff))

# make sequences
data=[]
target=[]
maxlen=20
length_of_sequence=len(diff)
for i in range(0,length_of_sequence-maxlen):
    data.append(diff[i:i+maxlen])
    target.append(diff[i+maxlen])
x=np.array(data).reshape(len(data),maxlen).astype('float32')
t=np.array(target).reshape(len(data),1).astype('float32')

from sklearn.model_selection import train_test_split
x_train_val,x_test,t_train_val,t_test=train_test_split(x,t,test_size=0.1,shuffle=False)
x_train,x_val,t_train,t_val=train_test_split(x_train_val,t_train_val,test_size=0.1,shuffle=False)


from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_t=StandardScaler()
sc_x.fit(x_train_val)
x_train_std=sc_x.transform(x_train)
x_val_std=sc_x.transform(x_val)
x_test_std=sc_x.transform(x_test)
sc_t.fit(t_train_val)
t_train_std=sc_t.transform(t_train)
t_val_std=sc_t.transform(t_val)
t_test_std=sc_t.transform(t_test)
#%%
x_train_std = x_train_std[:, :,np.newaxis].astype('float32')
x_val_std   =   x_val_std[:, :,np.newaxis].astype('float32')
t_train_std = t_train_std[:,np.newaxis].astype('float32')
t_val_std   =   t_val_std[:,np.newaxis].astype('float32')
x_test_std=x_test_std[:, :,np.newaxis].astype('float32')
t_test_std=t_test_std[:,np.newaxis].astype('float32')

print(x_train_std.shape,t_train_std.shape)
#%%
import numpy as np
x=[(1,2),(3,4)]
a=[]
for i in x:

    a.append(i[0])
print(a)
#%%
a=[1,2,3]
b=[1,3,4]
set(a) & set(b)
set(a) | set(b)