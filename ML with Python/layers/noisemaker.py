#%%
import matplotlib.pyplot as plt
import numpy as np

def gate(a):
    np.random.seed(a)
    x_xor=np.random.randn(200,2)
    y_xor=np.logical_xor(x_xor[:,0]>0,x_xor[:,1]>0)
    y_xor=np.where(y_xor,1,-1)
    print(x_xor)
    print(y_xor)
    plt.scatter(x_xor[y_xor==1,0],x_xor[y_xor==1,1],c='b',marker='x',label='1')

    plt.scatter(x_xor[y_xor==-1,0],x_xor[y_xor==-1,1],c='r',marker='s',label='-1')

    plt.xlim([-3,3])
    plt.ylim([-3,3])
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

gate(a=1) 