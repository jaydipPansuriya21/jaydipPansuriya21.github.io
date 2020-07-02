#!/usr/bin/env python
# coding: utf-8

# In[33]:


import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings 
import sklearn
warnings.filterwarnings("ignore", category=DeprecationWarning)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


train = pd.read_csv('Train/twitter_train_AV.csv')
test = pd.read_csv('Test/twitter_test_AV.csv')


# In[6]:


train.head()


# In[7]:


test.head()


# In[8]:


# user defined method to remove unwanted pattern in our tweets. takes two arguments one is original text and other is pattern we dont want.
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt


# In[9]:


# remove twitter handles (@user)
train['tweet'] = np.vectorize(remove_pattern)(train['tweet'], "@[\w]*")

test['tweet'] = np.vectorize(remove_pattern)(test['tweet'], "@[\w]*")


# In[10]:


train.head()


# In[11]:


test.head()


# In[12]:


# remove special characters, numbers, punctuations
train['tweet'] = train['tweet'].str.replace("[^a-zA-Z#]", " ")

test['tweet'] = test['tweet'].str.replace("[^a-zA-Z#]", " ")
train.head()
#test.head()


# In[13]:


# remove short words.
train['tweet'] = train['tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

test['tweet'] = test['tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

train.head()


# In[14]:


test.head()


# In[15]:


#tokenixation 
tokenized_train = train['tweet'].apply(lambda x: x.split())

tokenized_test = test['tweet'].apply(lambda x: x.split())

tokenized_train.head()


# In[16]:


#Stemming is a rule-based process of stripping the suffixes (“ing”, “ly”, “es”, “s” etc) from a word. 
#For example, For example – “play”, “player”, “played”, “plays” and “playing” are the different variations of the word – “play”.

from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenized_train = tokenized_train.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
tokenized_train.head()


# In[17]:


# Stemming for test data

tokenized_test = tokenized_test.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
tokenized_test.head()


# In[18]:


# now joint this tokens together.
for i in range(len(tokenized_train)):
    tokenized_train[i] = ' '.join(tokenized_train[i])

train['tweet'] = tokenized_train


# In[19]:


train.head()


# In[20]:


# same for test data
for i in range(len(tokenized_test)):
    tokenized_test[i] = ' '.join(tokenized_test[i])

test['tweet'] = tokenized_test


# In[21]:


test.head()


# In[22]:


#Understanding the common words used in the tweets: WordCloud (train data)
all_words = ' '.join([text for text in train['tweet']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[23]:


# Words in non racist/sexist tweets (train data)
normal_words = ' '.join([text for text in train['tweet'][train['label']==0]])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[24]:


negative_words = ' '.join([text for text in train['tweet'][train['label']==1]])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[25]:


# Understanding the impact of Hashtags on tweets sentiment

# function to collect hashtags
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags


# In[26]:


# extracting hashtags from non racist/sexist tweets

HT_regular = hashtag_extract(train['tweet'][train['label'] == 0])

# extracting hashtags from racist/sexist tweets
HT_negative = hashtag_extract(train['tweet'][train['label'] == 1])

# unnesting list
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])


# In[27]:


# Lets check hashtags in Non-Racist/Sexist Tweets
a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()


# In[28]:


# Lets check hashtags in Racist/Sexist Tweets
b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})
# selecting top 10 most frequent hashtags
e = e.nlargest(columns="Count", n = 10)   
plt.figure(figsize=(16,5))
ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()


# In[30]:


test.head()


# In[54]:


# Extracting Features from Cleaned Tweets (Bag of Words Features)
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# bag-of-words feature matrix
bow_train = bow_vectorizer.fit_transform(train['tweet'])
bow_test = bow_vectorizer.fit_transform(test['tweet'])


# In[95]:


print(bow_test)


# In[55]:


print(bow[:5])


# In[56]:


# Building model using TF-IDF features
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# TF-IDF feature matrix
tfidf_train = tfidf_vectorizer.fit_transform(train['tweet'])
tfidf_test = tfidf_vectorizer.fit_transform(test['tweet'])


# In[58]:


print(tfidf_test[:5])


# In[61]:


# splitting data into training and validation set
from sklearn.model_selection import train_test_split
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(bow_train, train['label'], random_state=42, test_size=0.3)

xtrain_tfidf = tfidf_train[ytrain.index]
xvalid_tfidf = tfidf_train[yvalid.index]


# In[79]:


from sklearn.preprocessing import LabelEncoder
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(ytrain)
Test_Y = Encoder.fit_transform(yvalid)


# In[84]:


# model training and prediction
from sklearn import model_selection, naive_bayes, svm

# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(xtrain_tfidf,Train_Y)
# predict the labels on validation dataset
predictions_NB = Naive.predict(xvalid_tfidf)


# In[88]:


from sklearn.metrics import accuracy_score
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)

