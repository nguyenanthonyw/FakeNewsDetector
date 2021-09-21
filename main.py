# Description
# This program detects fake news


import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('Fake.csv')

print(df.head)

df.drop_duplicates(inplace= True) #detect any duplicate
df.shape

df.isnull().sum()

df.dropna(axis=0, inplace= True) #drop any data with missing field
df.shape

#combines columns

df['combined'] =df['title'] + ' '+ df['subject']
print(df.head)

nltk.download('stopwords')



def process_text(text):
   nopunc = [char for char in text if char not in string.punctuation]
   nopunc = ''.join(nopunc)

   clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

   return clean_words


print(df['combined'].head().apply(process_text))

message_bow = CountVectorizer(analyzer=process_text).fit_transform(df['combined'])
x_train,x_test,y_train, y_text = train_test_split(message_bow, df['label'], test_size=.20, random_state= 0)

message_bow.shape

classifer = MultinomialNB
classifer.fit(x_train,y_train)

