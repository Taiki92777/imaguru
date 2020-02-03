#%%
import numpy as np

def sin(x,T=100):
    return np.sin(2*np.pi*x/T)
def toy_problem(T=100,ampl=0.05):
    x=np.arange(0,2*T+1)
    noise=ampl*np.random.uniform(low=-1,high=1,size=len(x))
    return sin(x)+noise

T=100
f=toy_problem(T)

length_of_sequences=2*T
maxlen=25

data=[]
target=[]
for i in range(0,length_of_sequences-maxlen+1):
    data.append(f[i:i+maxlen])
    target.append(f[i+maxlen])


x=np.array(data).reshape(len(data),maxlen,1)
y=np.array(target).reshape(len(data),1)
#%%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)
#%%
import matplotlib.pyplot as plt
from layers.classes import RNN_simple
rnn=RNN_simple(hout_size=10,epochs=3000,seed=0,eta=0.005)
x_train.reshape(158,25)
y_train.reshape(1,158)
rnn.fit(x_train,y_train)
rnn.lost_graph()