#%%
# probability mass function
from scipy.misc import comb
import math
def ensemble_error(n_classifier,error):
    k_start=int(math.ceil(n_classifier/2))
    probs=[comb(n_classifier,k)*error**k*(1-error)**(n_classifier-k) for k in range(k_start,n_classifier+1)]
    return sum(probs)

ensemble_error(n_classifier=11,error=0.25)
#%%
# plot error & ensemble_error relation
import numpy as np
import matplotlib.pyplot as plt
error_range=np.arange(0.0,1.01,0.01)
ens_errors=[ensemble_error(n_classifier=11,error=error) for error in error_range]

fig,ax=plt.subplots()
fig.set_facecolor('white')
ax.plot(error_range,ens_errors,label='Ensemble error',linewidth=2)
ax.plot(error_range,error_range,linestyle='--',label='Base error',linewidth=2)
plt.xlabel('Base error')
plt.ylabel('Base/Ensemble error')
plt.legend(loc='upper left')
plt.grid(alpha=0.5)
plt.show()