
#%%
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from itertools import product

def plot_decision_regions(x,y,classifier,test_idx=None,resolution=0.02,pipe=False):
    markers=('s','x','o','^','v')
    colors=('red','blue','lightgreen','gray','cyan')
    cmap=ListedColormap(colors[:len(np.unique(y))])

    x1_min,x1_max=x[:,0].min()-1,x[:,0].max()+1
    x2_min,x2_max=x[:,1].min()-1,x[:,1].max()+1

    xx1,xx2=np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
    if pipe:
        classifier.fit(x,y)
    z=classifier.predict(np.c_[xx1.ravel(),xx2.ravel()])

    z=z.reshape(xx1.shape)

    plt.contourf(xx1,xx2,z,alpha=0.4,cmap=cmap)

    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y==cl,0],y=x[y==cl,1],alpha=0.6,c=cmap(idx),marker=markers[idx],label=cl,edgecolor='black')

    if test_idx:
        x_test,y_test=x[test_idx,:],y[test_idx]
        plt.scatter(x_test[:,0],x_test[:,1],c='',edgecolors='black',alpha=1.0,linewidth=1,marker='o',s=100,label='test set')

def plot_decision_regions_row_column(x,y,x1title,x2title,classifiers,clf_labels,nrows,ncols,figsize,resolution=0.02):
    markers=('s','x','o','^','v')
    colors=('red','blue','lightgreen','gray','cyan')
    cmap=ListedColormap(colors[:len(np.unique(y))])

    x1_min,x1_max=x[:,0].min()-1,x[:,0].max()+1
    x2_min,x2_max=x[:,1].min()-1,x[:,1].max()+1

    xx1,xx2=np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
    
    fig,ax=plt.subplots(nrows=nrows,ncols=ncols,sharex='col',sharey='row',figsize=figsize)
    fig.set_facecolor('white')
    for idx,clf,tt in zip(product(range(nrows),range(ncols)),classifiers,clf_labels):
        clf.fit(x,y)
        z=clf.predict(np.c_[xx1.ravel(),xx2.ravel()])
        z=z.reshape(xx1.shape)
        if nrows==1:
            ax[idx[1]].contourf(xx1,xx2,z,alpha=0.4,cmap=cmap)
            for i,cl in enumerate(np.unique(y)):
                ax[idx[1]].scatter(x=x[y==cl,0],y=x[y==cl,1],alpha=0.6,c=cmap(i),marker=markers[i],label=cl,edgecolor='black')
            ax[idx[1]].set_title(tt)
            ax[idx[1]].set_title(tt)
            ax[idx[1]].set_xlabel(x1title)
            ax[idx[1]].set_ylabel(x2title)
        elif  ncols==1:
            ax[idx[0]].contourf(xx1,xx2,z,alpha=0.4,cmap=cmap)
            for i,cl in enumerate(np.unique(y)):
                ax[idx[0]].scatter(x=x[y==cl,0],y=x[y==cl,1],alpha=0.6,c=cmap(i),marker=markers[i],label=cl,edgecolor='black')
            ax[idx[0]].set_title(tt)
            ax[idx[0]].set_title(tt)
            ax[idx[0]].set_xlabel(x1title)
            ax[idx[0]].set_ylabel(x2title)
       
        else:
            ax[idx[0],idx[1]].contourf(xx1,xx2,z,alpha=0.4,cmap=cmap)
            for i,cl in enumerate(np.unique(y)):
                ax[idx[0],idx[1]].scatter(x=x[y==cl,0],y=x[y==cl,1],alpha=0.6,c=cmap(i),marker=markers[i],label=cl,edgecolor='black')
            ax[idx[0],idx[1]].set_title(tt)
            ax[idx[0],idx[1]].set_xlabel(x1title)
            ax[idx[0],idx[1]].set_ylabel(x2title)
       
            
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    plt.tight_layout()
    
def lin_regplot(x,y,model):
    plt.scatter(x,y,c='steelblue',edgecolors='white',s=70)
    plt.plot(x,model.predict(x),color='black',lw=2)
    return