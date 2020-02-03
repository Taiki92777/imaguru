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
#%%
df=df.drop(['PT_TOKYO','GOLD_NY','PT_NY','USDJPY'],axis=1)
#%%
values=df.iloc[:,1].values
#%%
# 階差取得
def _diff(values):
    diff=values[1:]-values[:-1]
    return diff
diff=_diff(values)
#%%
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
#%%
from sklearn.model_selection import train_test_split
x_train_val,x_test,t_train_val,t_test=train_test_split(x,t,test_size=0.1,shuffle=False)
x_train,x_val,t_train,t_val=train_test_split(x_train_val,t_train_val,test_size=0.1,shuffle=False)

#%%
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
#%%
# make model
from chainer import Chain
import chainer.links as L

class RNN(Chain):
    def __init__(self,n_input,n_hidden,n_output):
        super(RNN,self).__init__()
        with self.init_scope():

            self.l1=L.LSTM(n_input,n_hidden)
            self.l2=L.LSTM(n_hidden,n_hidden)
            self.l3=L.LSTM(n_hidden,n_hidden)
            self.l4=L.Linear(n_hidden,n_output)
        

    def __call__(self, x):
        """
        # Param
        - x (Variable: (S, F))
        S: samples
        F: features

        # Return
        -   (Variable: (S, 1))
        """
        h1=self.l1(x)
        h2=self.l2(h1)
        h3=self.l3(h2)
        y=F.relu(self.l4(h3))
        return y
        
    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        self.l3.reset_state()
#%%
# make loss function
import chainer.functions as F
from chainer import reporter

class SumMSE_overtime(L.Classifier):
    def __init__(self,predictor):
        super(SumMSE_overtime,self).__init__(predictor,lossfun=F.mean_squared_error)
    
    def __call__(self,x_STF,y_STF):
        """
        # Param
        - X_STF (Variable: (S, T, F))
        - y_STF (Variable: (S, T, F))
        S: samples
        T: time_steps
        F: features

        # Return
        - loss (Variable: (1, ))
        """
        seq_len=x_STF.shape[0]

        # add losses
        loss=0
        for t in range(seq_len):
            pred=self.predictor(x_STF[t].reshape(1,-1,1))
            obs=y_STF[t]
            loss+=self.lossfun(pred,obs)
        loss/=seq_len
        reporter.report({'loss':loss},self)

        return loss
#%%
# make updator
from chainer import training, Variable

class UpdaterRNN(training.StandardUpdater):
    def __init__(self,itr_train,optimizer,device=-1):
        super(UpdaterRNN,self).__init__(itr_train,optimizer,device=device)
    
    def update_core(self):
        itr_train=self.get_iterator('main')
        optimizer=self.get_optimizer('main')

        batch=itr_train.__next__()
        x_STF,y_STF=chainer.dataset.concat_examples(batch,self.device)
        loss=optimizer.target(Variable(x_STF),Variable(y_STF))
        optimizer.target.cleargrads()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()
#%%
# run
from chainer.optimizers import RMSprop
from chainer.iterators import SerialIterator
from chainer.training import extensions

n_input=maxlen
n_hidden=5
n_output=1

net=SumMSE_overtime(RNN(n_input,n_hidden,n_output))
optimizer=RMSprop()
optimizer.setup(net)
ds_train=list(zip(x_train_std,t_train_std))
ds_val=list(zip(x_val_std,t_val_std))
itr_train=SerialIterator(ds_train,batch_size=10,shuffle=False)
itr_val=SerialIterator(ds_val,batch_size=10,shuffle=False,repeat=False)

updater=UpdaterRNN(itr_train,optimizer)

trainer=training.Trainer(updater,(30,'epoch'),out='results')

eval_model=net.copy()
eval_rnn=eval_model.predictor

trainer.extend(extensions.Evaluator(
            itr_val, eval_model, device=-1,
            eval_hook=lambda _: eval_rnn.reset_state()))
# other extensions
trainer.extend(extensions.LogReport())
trainer.extend(extensions.snapshot_object(net.predictor, 
                                           filename='model_epoch-{.updater.epoch}'))
trainer.extend(extensions.PrintReport(
                ['epoch','main/loss','validation/main/loss']
            ))
trainer.extend(extensions.ProgressBar(update_interval=100,bar_length=50))
trainer.run()
#%%
import pandas as pd
log = pd.read_json('results/log')
log.plot(y=['main/loss', 'validation/main/loss'],
         figsize=(15,10),
         grid=True)
