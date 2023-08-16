# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 13:08:30 2023

@author: Marcus
"""

import pandas as pd
import gensim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import numpy as np
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("Language Detection.csv")

encoder = LabelEncoder()

df["Target"] = encoder.fit_transform(df["Language"])

X_train, X_test, Y_train, Y_test = train_test_split(df["Text"], df["Target"], test_size =0.33)

def preprocess(text):
    tokens = gensim.utils.simple_preprocess(text)
    return [token for token in tokens]

X_train = [preprocess(text) for text in X_train]

model = gensim.models.Word2Vec(X_train, vector_size=100, window=5, min_count=2, workers=-1)

def get_embedding(text):
    embeddings = []
    for word in text:
        if word in model.wv:
            embeddings.append(model.wv[word])
    if len(embeddings) > 0:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(model.vector_size)
    
X_train_embeddings = [get_embedding(text) for text in X_train]

X_test = [preprocess(text) for text in X_test]
X_test_embeddings = [get_embedding(text) for text in X_test]

clf = LogisticRegression().fit(X_train_embeddings, Y_train)

predict = clf.predict(X_test_embeddings)

actual = list(Y_test)
predict = list(predict)
print(classification_report(actual, predict))
