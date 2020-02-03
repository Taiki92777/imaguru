#%%
import numpy as np
import re
from nltk.corpus import stopwords

stop=stopwords.words('english')
# 不必要な要素を消して、絵文字を最後につけ、stopwordsを消す
def tokenizer(text):
    text=re.sub('<[^>]*>','',text)
    emoticons=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text=(re.sub('[\W]+',' ',text.lower())+''.join(emoticons).replace('-',''))
    tokenized=[w for w in text.split() if w not in stop]
    return tokenized
# 1文章だけ返したあと一旦止める関数
def stream_docs(path):
    with open(path,'r',encoding='utf-8') as csv:
        next(csv)
        for line in csv:
            text,label=line[:-3],int(line[-2])
            yield text,label
# size指定分だけ読み込む関数
def get_minibatch(doc_stream,size):
    docs,y=[],[]
    try:
        for _ in range(size):
            text,label=next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None,None
    return docs,y
#%%
doc_stream=stream_docs(path='movie_data.csv')
x,y=get_minibatch(doc_stream,size=2000)
print(y)

#%%
# HashingVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

vect=HashingVectorizer(decode_error='ignore',n_features=2**21,preprocessor=None,tokenizer=tokenizer)
clf=SGDClassifier(loss='log',random_state=1,n_iter=1)
doc_stream=stream_docs(path='movie_data.csv')
#%%
# out-of-core learning
import pyprind

pbar=pyprind.ProgBar(45)
classes=np.array([0,1])
for _ in range(45):
    x_train,y_train=get_minibatch(doc_stream,size=1000)
    if not x_train:
        break
    x_train=vect.transform(x_train)
    clf.partial_fit(x_train,y_train,classes=classes)
    pbar.update()
#%%
x_test,y_test=get_minibatch(doc_stream,size=5000)
x_test=vect.transform(x_test)
print('Accuracy: %.3f'%clf.score(x_test,y_test))
#%%
clf=clf.partial_fit(x_test,y_test)
#%%
# model persistence
import pickle
import os

dest=os.path.join('movieclassifier','pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)
pickle.dump(stop,open(os.path.join(dest,'stopwords.pkl'),'wb'),protocol=4)
pickle.dump(clf,open(os.path.join(dest,'classifier.pkl'),'wb'),protocol=4)
