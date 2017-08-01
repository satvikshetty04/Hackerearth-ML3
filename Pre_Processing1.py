'''
Pre-Processing files.
'''

import sys
import datetime
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

# --------------------- Initializing variables

start_time = datetime.datetime.now()
print("Started at: " + str(start_time))


# --------------------- Loading datasets

train = pd.read_csv("Data\\train.csv", header=0)
test = pd.read_csv("Data\\test.csv", header=0)


# --------------------- Creating features from datetime

train['day'] = pd.to_datetime(train['datetime']).dt.day
train['weekday'] = pd.to_datetime(train['datetime']).dt.weekday
train['hour'] = pd.to_datetime(train['datetime']).dt.hour
train['minute'] = pd.to_datetime(train['datetime']).dt.minute
test['day'] = pd.to_datetime(test['datetime']).dt.day
test['weekday'] = pd.to_datetime(test['datetime']).dt.weekday
test['hour'] = pd.to_datetime(test['datetime']).dt.hour
test['minute'] = pd.to_datetime(test['datetime']).dt.minute


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

# Obtaining unique offer ids when browser and device is null
unique_offerid_train = list(set(train[train['devid'].isnull() & train['browserid'].isnull()]['offerid']))
unique_offerid_train = sorted(unique_offerid_train)

unique_offerid_test = list(set(test[test['devid'].isnull() & test['browserid'].isnull()]['offerid']))
unique_offerid_test = sorted(unique_offerid_test)

# Creating dictionaries for Offer IDs vs Device IDs
dictionary_train = {}
for i in unique_offerid_train:
    if len(train[train['offerid']==i]['devid'].mode()) == 0:
        dictionary_train[str(i)] = 'Mobile'
    else:
        dictionary_train[str(i)] = train[train['offerid']==i]['devid'].mode()[0]

dictionary_test = {}
for i in list(set(unique_offerid_test).difference(unique_offerid_train)):
    if len(train[train['offerid']==i]['devid'].mode()) == 0:
        dictionary_test[str(i)] = 'Mobile'
    else:
        dictionary_test[str(i)] = train[train['offerid']==i]['devid'].mode()[0]

# Setting Device ID when device & browser are null
train_temp = train[train['devid'].isnull()].copy()
train_temp['devid'] = train_temp['offerid'].map(lambda x: dictionary_train[str(x)])
train[train['devid'].isnull()] = train_temp
del train_temp

test_temp = test[test['devid'].isnull()].copy()
test_temp['devid'] = test_temp['offerid'].map(lambda x: dictionary_test[str(x)]
                if dictionary_test.__contains__(str(x)) else dictionary_train[str(x)])
test[test['devid'].isnull()] = test_temp
del test_temp

# Checking number of null values now
train.isnull().sum(axis=0)/train.shape[0]
test.isnull().sum(axis=0)/test.shape[0]

print("Time taken: " + str(datetime.datetime.now() - start_time))


# --------------------- Replacing null values - Browser ID

train['browserid'] = np.where(train['browserid'].isnull()
                        &(train['devid']=='Mobile'),
                        'Firefox', train['browserid'] )

train['browserid'] = np.where(train['browserid'].isnull()
                        &(train['devid']=='Desktop'),
                        'Mozilla', train['browserid'] )

train['browserid'] = np.where(train['browserid'].isnull()
                        &(train['devid']=='Tablet'),
                        'Edge', train['browserid'] )

# Mozilla <---- Mozilla Firefox
train['browserid'] = np.where(train['browserid']=='Mozilla Firefox',
                        'Mozilla', train['browserid'] )


test['browserid'] = np.where(test['browserid'].isnull()
                        &(test['devid']=='Mobile'),
                        'Firefox', test['browserid'] )

test['browserid'] = np.where(test['browserid'].isnull()
                        &(test['devid']=='Desktop'),
                        'Mozilla', test['browserid'] )

test['browserid'] = np.where(test['browserid'].isnull()
                        &(test['devid']=='Tablet'),
                        'Edge', test['browserid'] )

# Mozilla <---- Mozilla Firefox
test['browserid'] = np.where(test['browserid']=='Mozilla Firefox',
                        'Mozilla', test['browserid'] )


# --------------------- Label Encoding

encode_cols = ['countrycode', 'browserid', 'devid']
for col in encode_cols:
    le = LabelEncoder()
    le.fit(list(train[col].values) + list(test[col].values))
    train[col] = le.transform(list(train[col]))
    test[col] = le.transform(list(test[col]))


# --------------------- Reordering columns

cols = train.columns.tolist()
cols.insert(len(cols), cols.pop(cols.index('click')))
train = train.reindex(columns=cols)


# --------------------- Saving files

train.to_csv("Data\\train_pp1.csv", index = False)
test.to_csv("Data\\test_pp1.csv", index = False)


print("Overall time taken: " + str(datetime.datetime.now() - start_time))