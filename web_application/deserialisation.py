#%%
# deserialisation
import pickle
import re
import os
import sys
# 同じ階層でもインポートできるとは限らないので対策が必要
sys.path.append(rf'C:\Users\taiki\OneDrive\デスクトップ\machinelearning\movieclassifier')
from vectorizer import vect

clf=pickle.load(open(os.path.join('movieclassifier','pkl_objects','classifier.pkl'),'rb'))
#%%
import numpy as np
label={0:'negative',1:'positive'}
example=['no deficts']
x=vect.transform(example)
print('Prediction: %s\nProbability: %.2f%%'%(label[clf.predict(x)[0]],np.max(clf.predict_proba(x))*100))
