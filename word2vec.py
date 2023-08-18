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
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("Language Detection.csv")

encoder = LabelEncoder()

df["Target"] = encoder.fit_transform(df["Language"])

df["clean"] = df["Text"].apply(lambda x: gensim.utils.simple_preprocess(x))

X_train, X_test, Y_train, Y_test = train_test_split(df["clean"], df["Target"], test_size =0.1)

model = gensim.models.Word2Vec(X_train, vector_size=100, window=5, min_count=2, workers=-1)

words = set(model.wv.index_to_key)

X_train_vect = np.array([np.array([model.wv[i] for i in ls if i in words]) for ls in X_train])
X_test_vect = np.array([np.array([model.wv[i] for i in ls if i in words]) for ls in X_test])

X_train_vect_avg = []
for v in X_train_vect:
    if v.size:
        X_train_vect_avg.append(v.mean(axis=0))
    else:
        X_train_vect_avg.append(np.zeros(100, dtype = float))
        
X_test_vect_avg = []
for v in X_test_vect:
    if v.size:
        X_test_vect_avg.append(v.mean(axis=0))
    else:
        X_test_vect_avg.append(np.zeros(100, dtype = float))

clf = RandomForestClassifier().fit(X_train_vect_avg, Y_train)

predict = clf.predict(X_test_vect_avg)

actual = list(Y_test)
predict = list(predict)
print(classification_report(actual, predict))
