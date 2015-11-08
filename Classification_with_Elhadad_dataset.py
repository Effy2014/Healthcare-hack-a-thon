
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import csv
import string
from nltk import NaiveBayesClassifier
from nltk import classify
from nltk.stem.lancaster import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack
from nltk.corpus import names
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA


# In[2]:

label2 = pd.read_csv("/Users/XW/Desktop/w3phi-2015/reuters/all_reuters_matched_articles_filtered.csv", header=None)
label1 = pd.read_csv("/Users/XW/Desktop/w3phi-2015/reuters/all_reuters_article_info.csv", header=None)
label2 = label2.drop([0,1], axis=1)
label1 = label1.drop([0], axis=1)


# In[3]:

label1.columns = [1,2,3,4,5,6]
label2.columns = [1,2,3,4,5,6]
#add label
label1['label'] = 1
label2['label'] = 0

data = pd.concat([label1,label2], ignore_index=True)
data = data.drop_duplicates()


sw = set(stopwords.words('english'))
punctuation = set(string.punctuation)
all_names = set([name.lower() for name in names.words()])


# In[4]:

def isStopWord(word):
	return (word in sw or word in punctuation) or not word.isalpha() or word in all_names



def tfidf(prefix,number,data):
	List = []
	for obj in data[number]:
	    temp = [ ]
	    for flag in obj.split(" "):
	        if not isStopWord(flag.lower()) and len(flag) > 1:
	            temp.append(flag)
	    List.append(temp)
	st = LancasterStemmer()
	NewList = []
	for L in List:
	    temp = []
	    for l in L:
	        temp.append(st.stem(l))
	        ll = " ".join(temp)
	    NewList.append(ll)
	vectorizer = TfidfVectorizer(stop_words='english')
	temp = vectorizer.fit_transform(NewList)
	temp = temp.todense()
	name = [prefix+column for column in list(vectorizer.vocabulary_)]
	output = pd.DataFrame(temp,columns=name)
	return(output)


# In[5]:

MESH = tfidf("MESH",6,data)
Abst = tfidf("Abst",5,data)
Cita = tfidf("Cita",1,data)


# In[40]:

final = pd.concat([MESH,Abst,Cita,data['label'].reset_index(drop=True)], axis=1, ignore_index = True)


# In[10]:

from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation


# In[45]:

Label = final.iloc[:,-1]
FLAG = final.iloc[:,:-1]


# In[46]:

FLAG = np.array(FLAG)
Label = np.array(Label)
kf = cross_validation.KFold(len(FLAG), n_folds=2)
for train_index, test_index in kf:
    FLAG_train, FLAG_test = FLAG[train_index], FLAG[test_index]
    label_train, label_test = Label[train_index], Label[test_index]

clf_l1_LR = LogisticRegression(C=1, penalty='l1', tol=0.01)
clf_l1_LR.fit(FLAG_train, label_train)
clf_l1_LR.predict(FLAG_test)
error = sum(abs(clf_l1_LR.predict(FLAG_test)-Label[test_index]))


# In[52]:

np.sort(clf_l1_LR.coef_)


# In[47]:

error/len(FLAG)


# In[ ]:



