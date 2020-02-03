#%%
import numpy as np
from sklearn.datasets import load_iris
from renom_tda.topology import Topology
from renom_tda.lens import TSNE

iris=load_iris()
print(iris)
data,target=iris.data,iris.target
#%%
topology=Topology()
topology.load_data(data)

metric=None
lens=[TSNE(components=[0,1])]
topology.fit_transform(metric=metric,lens=lens)
topology.map(resolution=15,overlap=0.5,eps=0.1,min_samples=3)
topology.color(target,color_method='mode',color_type='rgb')
topology.save('iris_TSNE_topology.png',fig_size=(10,10),node_size=10,edge_width=2)