#%%
# What is BoW model?
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

count=CountVectorizer()
docs=np.array(['The sun is shining','The weather is sweet','The sun is shining, the weather is sweet, and one and one is two'])
bag=count.fit_transform(docs)

print(count.vocabulary_)
#%%
print(bag.toarray())
#%%
# TF-IDF 単語にたいする出現頻度を考慮した重み付け
from sklearn.feature_extraction.text import TfidfTransformer

tfidf=TfidfTransformer(use_idf=True,norm='l2',smooth_idf=True)
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())
#%%
# data cleansing  delete unnecessary terms
import re
def preprocessor(text):
    text=re.sub('<[^>]*>','',text)
    emoticons=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text=(re.sub('[\W]+',' ',text.lower())+''.join(emoticons).replace('-',''))
    return text
#%%
# validation
import pandas as pd

df=pd.read_csv('movie_data.csv',encoding='utf-8')
preprocessor(df.loc[0,'review'][-50:])
#%%
preprocessor("</a>This :) is :( a test :-)!")
#%%
# cleansing all text
df['review']=df['review'].apply(preprocessor)
#%%
# tokenize
def tokenizer(text):
    return text.split()

tokenizer('runners like running and thus they run')
#%%
# porter tokenize 単語を原形に変換
from nltk.stem.porter import PorterStemmer

porter=PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

tokenizer_porter('runners like running and thus they run')
#%%
# delete STOP words supplied by NLTK
import nltk
nltk.download('stopwords')
#%%
from nltk.corpus import stopwords
stop=stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')if w not in stop]
#%%
# classify texts by using LogisticRegression
# split data into training and test
x_train,x_test,y_train,y_test=df.loc[:25000,'review'].values,df.loc[25000:,'review'].values,df.loc[:25000,'sentiment'].values,df.loc[25000:,'sentiment'].values
#%%
# Gridsearch for logisticregression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None)
param_grid=[{'vect__ngram_range':[(1,1)],
             'vect__stop_words':[stop,None],
             'vect__tokenizer':[tokenizer,tokenizer_porter],
             'clf__penalty':['l1','l2'],
             'clf__C':[1.0,10.0,100.0]},
            {'vect__ngram_range':[(1,1)],
             'vect__stop_words':[stop,None],
             'vect__tokenizer':[tokenizer,tokenizer_porter],
             'vect__use_idf':[False],
             'vect__norm':[None],
             'clf__penalty':['l1','l2'],
             'clf__C':[1,10,100]}]
lr_tfidf=Pipeline([('vect',tfidf),('clf',LogisticRegression(random_state=0))])
gs_lr_tfidf=GridSearchCV(lr_tfidf,param_grid,scoring='accuracy',cv=5,verbose=1,n_jobs=1)
gs_lr_tfidf.fit(x_train,y_train)
#%%
# check accuracy
print('Best parameter set: %s'% gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' gs_lr_tfidf.best_score_)
clf=gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f'% clf.score(x_test,y_test))