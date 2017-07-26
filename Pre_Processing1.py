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

# SS
train = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 3\\train.csv",
                    header=0)

test = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 3\\test.csv",
                    header=0)

# GJ
# train = pd.read_csv("train.csv")
# test = pd.read_csv("test.csv")


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

# Checking number of null values now
train.isnull().sum(axis=0)/train.shape[0]

print("Time taken: " + str(datetime.datetime.now() - start_time))

dictionary = {}
for i in unique_offerid:
    if len(train[train['offerid']==i]['devid'].mode()) == 0:
        dictionary[str(i)] = 'Mobile'
    else:
        dictionary[str(i)] = train[train['offerid']==i]['devid'].mode()[0]

train['devid'] = np.where(train['devid'].isnull() & train['browserid'].isnull(),
                        dictionary[str(train['offerid'])], train['devid'] )

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
                        'Internet Explorer', train['browserid'] )
#Mozilla <---- Mozilla Firefox
train['browserid'] = np.where(train['browserid']=='Mozilla Firefox',
                        'Mozilla', train['browserid'] )


# --------------------- Label Encoding
encode_cols = ['countrycode', 'browserid', 'devid']
for col in encode_cols:
    le = LabelEncoder()
    le.fit(list(train[col].values) + list(test[col].values))
    train[col] = le.transform(list(train[col]))
    test[col] = le.transform(list(test[col]))


# --------------------- Reordering columns
cols = train.columns.tolist()
print(cols)
cols.insert(len(cols), cols.pop(cols.index('click')))
train = train.reindex(columns=cols)


# --------------------- Saving files
train.to_csv("train_pp1.csv", index = False)
test.to_csv("test_pp1.csv", index = False)