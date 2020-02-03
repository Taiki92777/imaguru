#%%
import numpy as np
from sklearn.datasets import fetch_openml
from renom_tda.topology import Topology
from renom_tda.lens import PCA

data_path="../dataset"
data,target=fetch_openml('mnist_784', version=1, return_X_y=True)
data,target=data[::10],target[::10]
data=data.astype(np.float32)
data /=data.max()
#%%
topology=Topology()
topology.load_data(data)
#%%
metric=None
lens=[PCA(components=[0,1])]
topology.fit_transform(metric=metric,lens=lens)
#%%
topology.map(resolution=50,overlap=1,eps=0.4,min_samples=2)
#%%
topology.color(target,color_method='mode',color_type='rgb',normalize=True)
#%%
topology.save('mnist_pca_topology.png',fig_size=(15,15),node_size=1,edge_width=0.1,mode='spring',strength=0.05)
