'''
Predicting Missing SiteID
'''


import sys
import datetime
import numpy as np
import pandas as pd
import os
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

n = 10
isinstance(n,int)



# --------------------- Loading datasets

train = pd.read_csv("Data\\train_pp1.csv", header=0)
test = pd.read_csv("Data\\test_pp1.csv", header=0)
train.drop(['ID', 'click', 'datetime'],axis=1, inplace=True)
test.drop(['ID', 'datetime'],axis=1, inplace=True)

test_mod = train[train['siteid'].isnull()]
train_mod = train[~train['siteid'].isnull()]

test_mod.drop(['siteid'],axis=1, inplace=True)
np.random.seed(10)
rows = np.random.choice(train_mod.index.values, 1000000)
sampled_train = train_mod.loc[rows]

Y = sampled_train['siteid'].astype(int)
sampled_train.drop(['siteid'],axis=1, inplace=True)
X = sampled_train
rfcl = RandomForestClassifier(n_estimators=1, n_jobs=-1, random_state=125)
rfmod = rfcl.fit(X, Y)

xgbclassifier = XGBClassifier(n_estimators=10, nthread=-1, silent=False, seed=125)
# xgbmodel = xgbclassifier.fit(X_train, y_train)
xgbmodel = xgbclassifier.fit(X, Y)

