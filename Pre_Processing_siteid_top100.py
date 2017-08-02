# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 00:39:55 2017

@author: Govardhan
"""

'''
Pre-Processing files.
'''

import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier

# --------------------- Initializing variables

start_time = datetime.datetime.now()
print("Started at: " + str(start_time))


# --------------------- Loading datasets

train_real = pd.read_csv("Data\\train_pp2.csv", header=0)

train = train_real.copy()

# --------------------- Creating weekday and minute
train['weekday'] = pd.to_datetime(train['datetime']).dt.weekday
train['minute'] = pd.to_datetime(train['datetime']).dt.minute


# Mozilla <---- Mozilla Firefox
train['browserid'] = np.where(train['browserid']=='Mozilla Firefox',
                        'Mozilla', train['browserid'] )


# --------------------- Dropping ID, DateTime
train.drop(['ID','datetime'], axis = 1, inplace = True)

# --------------------- Generating top 1000 frequent siteids
siteidcounts = pd.DataFrame(train.groupby('siteid').size().rename('counts'))
sortSiteid = siteidcounts.sort_values('counts', ascending = False)
listTopSiteids = sortSiteid.index[0:299].astype(int).tolist()

trainTopSiteid = train[~train['siteid'].isnull()&train['siteid'].isin(listTopSiteids)].copy()
testTopSiteid = train[train['siteid'].isnull()].copy()

trainTopSiteid.isnull().sum(axis=0)/trainTopSiteid.shape[0]

# --------------------- this code is from ML3 website.
#cols = ['siteid','offerid','category','merchant']
#for x in cols:
#    trainTopSiteid[x] = trainTopSiteid[x].astype('object')
#    testTopSiteid[x] = testTopSiteid[x].astype('object')


# --------------------- Label Encoding

encode_cols = ['countrycode', 'browserid', 'devid']
for col in encode_cols:
    le = LabelEncoder()
    le.fit(list(trainTopSiteid[col].values) + list(testTopSiteid[col].values))
    trainTopSiteid[col] = le.transform(list(trainTopSiteid[col]))
    testTopSiteid[col] = le.transform(list(testTopSiteid[col]))

# siteidcol = 'siteid'
# le1 = LabelEncoder()
# le1.fit(list(trainTopSiteid[siteidcol].values))
# trainTopSiteid[siteidcol] = le1.transform(list(trainTopSiteid[siteidcol]))



# --------------------- Reordering columns

cols = trainTopSiteid.columns.tolist()
cols.insert(len(cols), cols.pop(cols.index('siteid')))
trainTopSiteid = trainTopSiteid.reindex(columns=cols)
testTopSiteid = testTopSiteid.reindex(columns=cols)

# --------------------- Saving files

# trainTopSiteid.to_csv("Data\\train_pp_siteid_top1000.csv", index = False)

np.random.seed(1000)
#rows = np.random.choice(trainTopSiteid.index.values, 1000000)
#sampled_train = trainTopSiteid.loc[rows]

trainX = trainTopSiteid
trainY = trainTopSiteid['siteid']
trainX.drop(['siteid'], axis = 1, inplace = True)

X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size = 0.5)
# model_train_siteid = CatBoostClassifier(depth=15, iterations=5, learning_rate=0.1, eval_metric='AUC',
#  classes_count = 999, random_seed=1, calc_feature_importance=True, verbose=True)
# del sampled_train
del train
del trainTopSiteid
# del rows
del trainX
del trainY
del listTopSiteids
#
# cat_cols = [0,1,2,3,4,5,6,9,10]
#
# model_train_siteid.fit(X_train
#           ,y_train
#           ,cat_features=cat_cols
#           ,eval_set = (X_test, y_test)
#           ,use_best_model = True
#          )

model_train_siteid = XGBClassifier(n_estimators=10, nthread=-1, silent=False, seed=125, learning_rate=0.2)
model_train_siteid.fit(X_train,y_train)
model_train_siteid.score(X_test,y_test)


#Remove siteid from testTopSiteid
# testTopSiteid.drop(['siteid'], axis = 1, inplace = True)
pred = model_train_siteid.predict(testTopSiteid)
# predSiteid = le1.inverse_transform(pred.astype('int'))
testTopSiteid = train_real[train_real['siteid'].isnull()].copy()
testTopSiteid['siteid'] = pred
train_real[train_real['siteid'].isnull()] = testTopSiteid
train_real.to_csv('data\\train_br_dev_site_pp4.csv',index=False)











####################################################
#
# --------------------------- For Test Data
#
####################################################

# --------------------- Loading datasets

start_time = datetime.datetime.now()
print("Started at: " + str(start_time))


# --------------------- Loading datasets

test_real = pd.read_csv("Data\\test_br_dev_pp3.csv", header=0)

test = test_real.copy()

# --------------------- Creating weekday and minute
test['weekday'] = pd.to_datetime(test['datetime']).dt.weekday
test['minute'] = pd.to_datetime(test['datetime']).dt.minute


# Mozilla <---- Mozilla Firefox
test['browserid'] = np.where(test['browserid']=='Mozilla Firefox',
                        'Mozilla', test['browserid'] )


# --------------------- Dropping ID, DateTime
test.drop(['ID','datetime'], axis = 1, inplace = True)

# --------------------- Generating top 1000 frequent siteids
siteidcounts = pd.DataFrame(test.groupby('siteid').size().rename('counts'))
sortSiteid = siteidcounts.sort_values('counts', ascending = False)
listTopSiteids = sortSiteid.index[0:999].astype(int).tolist()

testTopSiteid = test[~test['siteid'].isnull()&test['siteid'].isin(listTopSiteids)]
testtestTopSiteid = test[test['siteid'].isnull()]

testTopSiteid.isnull().sum(axis=0)/testTopSiteid.shape[0]

# --------------------- this code is from ML3 website.
cols = ['siteid','offerid','category','merchant']
for x in cols:
    testTopSiteid[x] = testTopSiteid[x].astype('object')
    testtestTopSiteid[x] = testtestTopSiteid[x].astype('object')


# --------------------- Label Encoding

encode_cols = ['countrycode', 'browserid', 'devid']
for col in encode_cols:
    le = LabelEncoder()
    le.fit(list(testTopSiteid[col].values) + list(testtestTopSiteid[col].values))
    testTopSiteid[col] = le.transform(list(testTopSiteid[col]))
    testtestTopSiteid[col] = le.transform(list(testtestTopSiteid[col]))

siteidcol = 'siteid'
le = LabelEncoder()
le.fit(list(testTopSiteid[siteidcol].values))
testTopSiteid[siteidcol] = le.transform(list(testTopSiteid[siteidcol]))
    
    
# --------------------- Reordering columns

cols = testTopSiteid.columns.tolist()
cols.insert(len(cols), cols.pop(cols.index('siteid')))
testTopSiteid = testTopSiteid.reindex(columns=cols)
testtestTopSiteid = testtestTopSiteid.reindex(columns=cols)

# --------------------- Saving files

# testTopSiteid.to_csv("Data\\test_pp_siteid_top1000.csv", index = False)

np.random.seed(1000)
#rows = np.random.choice(testTopSiteid.index.values, 1000000)
#sampled_test = testTopSiteid.loc[rows]

#testX = sampled_test
testX = testTopSiteid
testY = testTopSiteid['siteid']
#testY = sampled_test['siteid']
testX.drop(['siteid'], axis = 1, inplace = True)

X_test, X_testtest, y_test, y_testtest = train_test_split(testX, testY, test_size = 0.4)
model_test_siteid = CatBoostClassifier(depth=15, iterations=15, learning_rate=0.1, eval_metric='AUC', random_seed=1, calc_feature_importance=True, verbose=True)

del sampled_test
del test
del testTopSiteid
del rows
del testX
del testY
del listTopSiteids

#Give proper col nums. click shouldn't be there
cat_cols = [0,1,2,3,4,5,6,9,1000]

model_test_siteid.fit(X_test
          ,y_test
          ,cat_features=cat_cols
          ,eval_set = (X_testtest, y_testtest)
          ,use_best_model = True
         )

#Remove siteid from testTopSiteid
testtestTopSiteid.drop(['siteid'], axis = 1, inplace = True)
pred = model_test_siteid.predict(testtestTopSiteid)
predSiteid = le.inverse_transform(pred.astype('int'))
test_real[test_real['siteid'].isnull()]['siteid'] = predSiteid

test_real.to_csv('data\\test_br_dev_site_pp4.csv',index=False)