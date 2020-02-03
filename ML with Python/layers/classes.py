import numpy as np
from numpy.random import seed
from sklearn.base import clone
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import operator
import matplotlib.pyplot as plt


class Perceptron1(object):
    def __init__(self,eta,n_iter,random_state=1):
        self.eta=eta
        self.n_iter=n_iter
        self.random_state=random_state

    def fit(self,x,y):
        rgen=np.random.RandomState(self.random_state)
        self.w_=rgen.normal(loc=0.0,scale=0.01,size=1+x.shape[1])
        self.errors_=[]

        for _ in range(self.n_iter):
            errors=0
            for xi, target in zip(x,y):
                update=self.eta*(target-self.predict(xi))
                self.w_[1:]+=update*xi
                self.w_[0]+=update
                errors+=int(update!=0.0)
            self.errors_.append(errors)
        return self

    def net_input(self,x):
        return np.dot(x,self.w_[1:])+self.w_[0]

    def predict(self,x):
        return np.where(self.net_input(x)>=0.0,1,-1)

class AdalineGD(object):
    #ADAptive LInear NEuron
    def __init__(self,eta,n_iter,random_state=1):
        self.eta=eta
        self.n_iter=n_iter
        self.random_state=random_state

    def fit(self,x,y):
        rgen=np.random.RandomState(self.random_state)
        self.w_=rgen.normal(loc=0.0,scale=0.01,size=1+x.shape[1])
        self.cost_=[]

        for i in range(self.n_iter):
            net_input=self.net_input(x)
            output=self.activation(net_input)
            errors=(y-output)
            self.w_[1:]+=self.eta*x.T.dot(errors)
            self.w_[0]+=self.eta*errors.sum()
            cost=(errors**2).sum()/2.0
            self.cost_.append(cost)
        return self

    def net_input(self,x):
        return np.dot(x,self.w_[1:])+self.w_[0]

    def activation(self,x):
        return x

    def predict(self,x):
        return np.where(self.activation(self.net_input(x))>=0.0,1,-1)

class AdalineGD_stochasticgradientdescent(object):

    def __init__(self,eta,n_iter,shuffle=True,random_state=None):
        self.eta=eta
        self.n_iter=n_iter
        self.w_initialized=False
        self.shuffle=shuffle
        self.random_state=random_state

    def fit(self,x,y):
        self._initialize_weights(x.shape[1])
        self.cost_=[]
        for i in range(self.n_iter):
            if self.shuffle:
                x,y=self._shuffle(x,y)
            
            cost=[]
            for xi, target in zip(x,y):
                cost.append(self._update_weights(xi,target))

            avg_cost=sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self,x,y):
        if not self.w_initialized:
            self._initialize_weights(x.shape[1])

        if y.ravel().shape.shape[0]>1:
            for xi,target in zip(x,y):
                self._update_weights(xi,target)
            
        else:
            self._update_weights(x,y)
        
        return self

    def _shuffle(self,x,y):
        r=self.rgen.permutation(len(y))
        return x[r],y[r]

    def _initialize_weights(self,m):
        self.rgen=np.random.RandomState(self.random_state)
        self.w_=self.rgen.normal(loc=0.0,scale=0.01,size=1+m)
        self.w_initialized=True

    def _update_weights(self,xi,target):
        output=self.activation(self.net_input(xi))
        error=(target-output)
        self.w_[1:]+=self.eta*xi.dot(error)
        self.w_[0]+=self.eta*error
        cost=0.5*error**2
        return cost

    def net_input(self,x):
        return np.dot(x,self.w_[1:])+self.w_[0]

    def activation(self,x):
        return x

    def predict(self,x):
        return np.where(self.activation(self.net_input(x))>=0.0,1,-1)

class LogisticRegressionGD(object):
    
    def __init__(self,eta,n_iter,random_state=1):
        self.eta=eta
        self.n_iter=n_iter
        self.random_state=random_state

    def fit(self,x,y):
        rgen=np.random.RandomState(self.random_state)
        self.w_=rgen.normal(loc=0.0,scale=0.01,size=1+x.shape[1])
        self.cost_=[]

        for i in range(self.n_iter):
            net_input=self.net_input(x)
            output=self.activation(net_input)
            log_likelihood=-np.dot(y,np.log(output))-np.dot(1-y,np.log(1-output))
            errors=(y-output)
            self.w_[1:]+=self.eta*x.T.dot(errors)
            self.w_[0]+=self.eta*errors.sum()
            self.cost_.append(log_likelihood)
        return self

    def net_input(self,x):
        return np.dot(x,self.w_[1:])+self.w_[0]

    def activation(self,x):
        return 1/(1+np.exp(-x))

    def predict(self,x):
        return np.where(self.activation(self.net_input(x))>=0.5,1,0)

#逐次特徴選択アルゴリズム　次元削減法の1つ（もう一つは特徴抽出）
#SBS(逐次後退選択)　過学習に陥いっているとき使うと予測性能の改善が見られる
'''SBSの手順
1 全特徴空間Xdの次元数dについて、k=dで初期化する
2 Jの評価を最大化する特徴量xiを決定(前後の性能低下が最小)　xi=argmax J(Xk-x)
3 特徴量集合から特徴量xiを削除　Xk-1:=Xk-xi ;k:=k-1
4 kが目的とする特徴量の個数に等しくなれば終了　or２に戻る
'''
class SBS():
    def __init__(self,estimator,k_features,scoring=accuracy_score,test_size=0.25,random_state=1):
        self.scoring=scoring #評価関数J
        self.estimator=clone(estimator) #推定器
        self.k_features=k_features
        self.test_size=test_size
        self.random_state=random_state

    def fit(self,x,y):
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=self.test_size,random_state=self.random_state)
        dim=x_train.shape[1] #特徴量の個数把握
        self.indices_=tuple(range(dim)) #全特徴量を示す列インデックスをタプルで把握
        self.subsets_=[self.indices_] #そのタプルをリストの[0]に格納
        #全特徴量から求めたスコア
        score=self._calc_score(x_train,y_train,x_test,y_test,self.indices_)
        self.scores_=[score]

        while dim>self.k_features:
            scores=[]
            subsets=[]
            #１つ特徴量が少ないレベルでのスコアと各組合せの列インデックスのリストを記憶
            for p in combinations(self.indices_,r=dim-1):
                #その特徴量の組み合わせでのスコア
                score=self._calc_score(x_train,y_train,x_test,y_test,p)
                scores.append(score)
                #その特徴量の組み合わせ（リスト）を格納
                subsets.append(p)
            #最良スコアのインデックスを抽出
            best=np.argmax(scores)
            #subsetsリストの最良スコアインデックスでのp（タプル）を抽出
            self.indices_=subsets[best]
            self.subsets_.append(self.indices_)
            dim-=1
            #スコア格納
            self.scores_.append(scores[best])
        
        #最後に格納したスコア
        self.k_scores_=self.scores_[-1]
        
        return self

    def transform(self,x):
        #サンプルデータの抽出特徴量を返す
        return x[:,self.indices_]

    def _calc_score(self,x_train,y_train,x_test,y_test,indices):
        self.estimator.fit(x_train[:,indices],y_train)
        y_pred=self.estimator.predict(x_test[:,indices])
        score=self.scoring(y_test,y_pred)
        return score
# 多数決アンサンブル学習器
class MajorityVoteClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self,classifiers,vote='classlabel',weights=None):
        self.classifiers=classifiers
        self.named_classifiers={key: value for key, value in _name_estimators(classifiers)}
        self.vote=vote
        self.weights=weights

    def fit(self,x,y):
        self.lablenc_=LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_=self.lablenc_.classes_
        self.classifiers_=[]
        for clf in self.classifiers:
            fitted_clf=clone(clf).fit(x,self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self,x):
        if self.vote=='probability':
            maj_vote=np.argmax(self.predict_proba(x),axis=1)

        else:
            predictions=np.asarray([clf.predict(x) for clf in self.classifiers_]).T
            maj_vote=np.apply_along_axis(lambda x:np.argmax(np.bincount(x,weights=self.weights)),axis=1,arr=predictions)

        maj_vote=self.lablenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self,x):
        probas=np.asarray([clf.predict_proba(x) for clf in self.classifiers_])
        avg_proba=np.average(probas,axis=0,weights=self.weights)
        return avg_proba

    def get_params(self,deep=True):
        if not deep:
            return super(MajorityVoteClassifier,self).get_params(deep=False)

        else:
            out=self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s'%(name,key)]=value
            return out

class LinearRegressionGD(object):
    def __init__(self,eta=0.001,n_iter=20):
        self.eta=eta
        self.n_iter=n_iter

    def fit(self,x,y):
        self.w_=np.zeros(1+x.shape[1])
        self.cost_=[]
        for i in range(self.n_iter):
            output=self.net_input(x)
            errors=(y-output)
            self.w_[1:]+=self.eta*x.T.dot(errors)
            self.w_[0]+=self.eta*errors.sum()
            cost=(errors**2).sum()/2
            self.cost_.append(cost)
        return self
    
    def net_input(self,x):
        return np.dot(x,self.w_[1:])+self.w_[0]

    def predict(self,x):
        return self.net_input(x)

class RNN_simple(object):
    def __init__(self,hout_size,epochs,seed,eta=0.001):
        self.hout_size=hout_size
        self.epochs=epochs
        self.eta=eta
        self.seed=seed
        
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def fit(self,x,y):
        self.rgen=np.random.RandomState(seed=self.seed)
        self.w_xh=self.rgen.randn(x.shape[1],self.hout_size)
        self.w_hh=self.rgen.randn(self.hout_size,self.hout_size)
        self.b_h=self.rgen.randn(1,self.hout_size)
        self.w_hy=self.rgen.randn(self.hout_size,1)
        self.b_y=self.rgen.randn(1,1)
        self.lost_func=[]
        for _ in range(self.epochs):
            hidden_output=[]
            y_input=[]
            y_output=[]
        
            hidden_output.append(np.dot(x[0,:].T,self.w_xh)+self.b_h)
            y_input.append(self.sigmoid(hidden_output[0]))
            y_output.append(np.dot(y_input[0],self.w_hy)+self.b_y)
            e_h=[]
            e_h.append((y_output[0]-y[0])*self.w_hy.T*(1-y_input[0])*y_input[0])
            for i in range(x.shape[0]-1):
                hidden_output.append(np.dot(x[i+1,:].T,self.w_xh)+np.dot(hidden_output[i],self.w_hh)+self.b_h)
                y_input.append(self.sigmoid(hidden_output[i+1]))
                y_output.append(self.sigmoid(np.dot(y_input[i+1],self.w_hy)+self.b_y))
                e_h.append((y_output[i+1]-y[i+1])*self.w_hy.T*(1-y_input[i+1])*y_input[i+1])
            hidden_output.insert(0,np.zeros(self.hout_size))
            del hidden_output[-1]
            for i in range(x.shape[0]-1)[::-1]:
                self.w_xh-=self.eta*(np.dot(x[i,:],np.array(e_h[i])))
                self.w_hh-=self.eta*(np.dot(np.array(hidden_output[i]).reshape(self.hout_size,1),np.array(e_h[i])))
                self.b_h-=self.eta*(np.array(e_h[i]))
            self.w_hy-=self.eta*((int(y_output[-1])-int(y[-1]))*np.array(hidden_output[i]).reshape(self.hout_size,1))
            self.b_y-=self.eta*((y_output[-1]-y[-1]))
            
            self.lost_func.append(np.sum((np.array(y_output)-y)**2)*0.5/x.shape[0])
    
    def lost_graph(self):
        plt.plot(np.arange(len(self.lost_func)),self.lost_func)
        plt.show()
