
import sys
import datetime
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# --------------------- Initializing variables

start_time = datetime.datetime.now()
print("Started at: " + str(start_time))


# --------------------- Loading datasets

train = pd.read_csv("Data\\train.csv", header=0)
test = pd.read_csv("Data\\test.csv", header=0)


# --------------------- Creating features from datetime

train['day'] = pd.to_datetime(train['datetime']).dt.day
train['hour'] = pd.to_datetime(train['datetime']).dt.hour
test['day'] = pd.to_datetime(test['datetime']).dt.day
test['hour'] = pd.to_datetime(test['datetime']).dt.hour
train['hour_range'] = np.where(train['hour'].isin([0,1,20,21,22,23]), 1, 0)
test['hour_range'] = np.where(test['hour'].isin([0,1,20,21,22,23]), 1, 0)


# --------------------- Replacing null values - Device ID

# Checking initial number of null values
train.isnull().sum(axis=0)/train.shape[0]
test.isnull().sum(axis=0)/test.shape[0]

train['devid'] = np.where(train['devid'].isnull()
                        &(~train['browserid'].isnull())
                        &(train['browserid'].isin(['IE', 'Google Chrome', 'Firefox', 'Opera'])),
                        'Mobile', train['devid'] )

train['devid'] = np.where(train['devid'].isnull()
                        &(~train['browserid'].isnull())
                        &(train['browserid'].isin(['InternetExplorer', 'Mozilla Firefox', 'Mozilla', 'Chrome'])),
                        'Desktop', train['devid'] )

train['devid'] = np.where(train['devid'].isnull()
                        &(~train['browserid'].isnull())
                        &(train['browserid'].isin(['Internet Explorer', 'Safari', 'Edge'])),
                        'Tablet', train['devid'] )

test['devid'] = np.where(test['devid'].isnull()
                        &(~test['browserid'].isnull())
                        &(test['browserid'].isin(['IE', 'Google Chrome', 'Firefox', 'Opera'])),
                        'Mobile', test['devid'] )

test['devid'] = np.where(test['devid'].isnull()
                        &(~test['browserid'].isnull())
                        &(test['browserid'].isin(['InternetExplorer', 'Mozilla Firefox', 'Mozilla', 'Chrome'])),
                        'Desktop', test['devid'] )

test['devid'] = np.where(test['devid'].isnull()
                        &(~test['browserid'].isnull())
                        &(test['browserid'].isin(['Internet Explorer', 'Safari', 'Edge'])),
                        'Tablet', test['devid'] )

train_temp = train.copy()
test_temp = test.copy()

# For training data
cols_to_drop = ['ID', 'datetime', 'siteid', 'browserid', 'click']
train_temp.drop(cols_to_drop, axis=1, inplace=True)
Y = train_temp[~train_temp['devid'].isnull()]['devid'].copy()
X = train_temp[~train_temp['devid'].isnull()].copy()
X.drop('devid', axis=1, inplace=True)
train_temp_test = train_temp[train_temp['devid'].isnull()].copy()
train_temp_test.drop('devid', axis=1, inplace=True)
encode_cols = ['countrycode']
for col in encode_cols:
    le = LabelEncoder()
    le.fit(list(X[col].values) + list(train_temp_test[col].values))
    X[col] = le.transform(list(X[col]))
    train_temp_test[col] = le.transform(list(train_temp_test[col]))
xgbclassifier = XGBClassifier(n_estimators=100, nthread=-1, silent=False, seed=125, learning_rate=0.2)
xgbmodel = xgbclassifier.fit(X, Y)
pred = xgbmodel.predict(train_temp_test)
train_temp = train[train['devid'].isnull()].copy()
train_temp['devid'] = pred
train[train['devid'].isnull()] = train_temp

# For testing data

cols_to_drop = ['ID', 'datetime', 'siteid', 'browserid']
test_temp.drop(cols_to_drop, axis=1, inplace=True)
Y = test_temp[~test_temp['devid'].isnull()]['devid'].copy()
X = test_temp[~test_temp['devid'].isnull()].copy()
X.drop('devid', axis=1, inplace=True)
test_temp_test = test_temp[test_temp['devid'].isnull()].copy()
test_temp_test.drop('devid', axis=1, inplace=True)
encode_cols = ['countrycode']
for col in encode_cols:
    le = LabelEncoder()
    le.fit(list(X[col].values) + list(test_temp_test[col].values))
    X[col] = le.transform(list(X[col]))
    test_temp_test[col] = le.transform(list(test_temp_test[col]))
xgbclassifier2 = XGBClassifier(n_estimators=100, nthread=-1, silent=False, seed=125, learning_rate=0.2)
xgbmodel2 = xgbclassifier2.fit(X, Y)
pred = xgbmodel2.predict(test_temp_test)
test_temp = test[test['devid'].isnull()].copy()
test_temp['devid'] = pred
test[test['devid'].isnull()] = test_temp


# --------------------- Replacing null values - Browser ID

# Checking initial number of null values
train.isnull().sum(axis=0)/train.shape[0]
test.isnull().sum(axis=0)/test.shape[0]

train_temp = train.copy()
test_temp = test.copy()

# For training data
cols_to_drop = ['ID', 'datetime', 'siteid', 'click']
train_temp.drop(cols_to_drop, axis=1, inplace=True)
Y = train_temp[~train_temp['browserid'].isnull()]['browserid'].copy()
X = train_temp[~train_temp['browserid'].isnull()].copy()
X.drop('browserid', axis=1, inplace=True)
train_temp_test = train_temp[train_temp['browserid'].isnull()].copy()
train_temp_test.drop('browserid', axis=1, inplace=True)
encode_cols = ['countrycode', 'devid']
for col in encode_cols:
    le = LabelEncoder()
    le.fit(list(X[col].values) + list(train_temp_test[col].values))
    X[col] = le.transform(list(X[col]))
    train_temp_test[col] = le.transform(list(train_temp_test[col]))
xgbclassifier = XGBClassifier(n_estimators=100, nthread=-1, silent=False, seed=125, learning_rate=0.2)
xgbmodel = xgbclassifier.fit(X, Y)
pred = xgbmodel.predict(train_temp_test)
train_temp = train[train['browserid'].isnull()].copy()
train_temp['browserid'] = pred
train[train['browserid'].isnull()] = train_temp

train.to_csv("Data\\train_pp2.csv", index = False)


# For testing data
cols_to_drop = ['ID', 'datetime', 'siteid']
test_temp.drop(cols_to_drop, axis=1, inplace=True)
Y = test_temp[~test_temp['browserid'].isnull()]['browserid'].copy()
X = test_temp[~test_temp['browserid'].isnull()].copy()
X.drop('browserid', axis=1, inplace=True)
test_temp_test = test_temp[test_temp['browserid'].isnull()].copy()
test_temp_test.drop('browserid', axis=1, inplace=True)
encode_cols = ['countrycode', 'devid']
for col in encode_cols:
    le = LabelEncoder()
    le.fit(list(X[col].values) + list(test_temp_test[col].values))
    X[col] = le.transform(list(X[col]))
    test_temp_test[col] = le.transform(list(test_temp_test[col]))
xgbclassifier2 = XGBClassifier(n_estimators=100, nthread=-1, silent=False, seed=125, learning_rate=0.2)
xgbmodel2 = xgbclassifier2.fit(X, Y)
pred = xgbmodel2.predict(test_temp_test)
test_temp = test[test['browserid'].isnull()].copy()
test_temp['browserid'] = pred
test[test['browserid'].isnull()] = test_temp

test.to_csv("Data\\test_pp2.csv", index = False)
