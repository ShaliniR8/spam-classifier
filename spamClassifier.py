# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd #manage dataframe
import string #string operation like removing punctuation
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score


#converting spam or not spam to spam=1 and ham=0
def spam_or_not(s):
    if s=="ham":
        return 0
    else:
        return 1

#
df=pd.read_csv('spam.csv',encoding='latin1')
df=df.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)
df=df.rename(columns={"v1":"class","v2":"text"})
df["class"]=df["class"].apply(spam_or_not)
df["length"]=df["text"].apply(len) #to make a column with header length which stores length of each text

#preprocessing
def pre_process(text):
    text=text.translate(str.maketrans('','',string.punctuation))
    text=[word for word in text.split() if word.lower() not in stopwords.words('english')]
    word=""
    for i in text:
        stemmer=SnowballStemmer("english")
        word+=stemmer.stem(i)+" "
    return word
    
text_process=df["text"].copy()
text_process=text_process.apply(pre_process)
vectorizer=TfidfVectorizer("english")
features=vectorizer.fit_transform(text_process)
X_train,X_test,y_train,y_test=train_test_split(features,df["class"],test_size=0.25,random_state=42)

lm=linear_model.LogisticRegression()
model=lm.fit(X_train,y_train)
predictions=lm.predict(X_test)

print (accuracy_score(y_test,predictions))