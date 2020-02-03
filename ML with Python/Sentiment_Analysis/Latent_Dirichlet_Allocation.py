#%%
# load data
import pandas as pd
df=pd.read_csv('movie_data.csv',encoding='utf-8')
#%%
# make BoW matrixes
from sklearn.feature_extraction.text import CountVectorizer
count=CountVectorizer(stop_words='english',max_df=0.1,max_features=5000)
x=count.fit_transform(df['review'].values)
#%%
# use LDA model to extract topics
from sklearn.decomposition import LatentDirichletAllocation
lda=LatentDirichletAllocation(n_topics=10,random_state=123,learning_method='batch')
x_topics=lda.fit_transform(x)
#%%
# show important words of each topics
import numpy as np
n_top_words=5
feature_names=count.get_feature_names()
for topic_idx, topic in enumerate(lda.components_):
    print('Topic %d:'%(topic_idx+1))
    print(' '.join([feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]]))
#%%
# show example 'horror'
horror=x_topics[:,5].argsort()[::-1]
for iter_idx,movie_idx in enumerate(horror[:3]):
    print('\nHorror movie #%d:'%(iter_idx+1))
    print(df['review'][movie_idx][:300],'...')