'''
Building a Model using XGBoost
'''

import sys
import datetime
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier

# --------------------- Initializing variables

start_time = datetime.datetime.now()
print("Started at: " + str(start_time))
pred_df = pd.DataFrame()

# --------------------- Loading datasets

train = pd.read_csv("Data\\train_pp1.csv", header=0)
test = pd.read_csv("Data\\test_pp1.csv", header=0)


# --------------------- Imputing siteid

train['siteid'].fillna(-999, inplace=True)
test['siteid'].fillna(-999, inplace=True)

# --------------------- Dropping columns

pred_df['ID'] = test['ID']
cols_to_drop = ['ID', 'datetime']
train.drop(cols_to_drop, axis=1, inplace=True)
test.drop(cols_to_drop, axis=1, inplace=True)


# --------------------- Splitting data

Y = train['click'].as_matrix()
train.drop('click', axis=1, inplace=True)
X = train.as_matrix()
test = test.as_matrix()


# --------------------- Creating model

xgbclassifier = XGBClassifier(n_estimators=100, nthread=-1, silent=False, seed=125)
xgbmodel = xgbclassifier.fit(X, Y)
# pred = xgbmodel.predict(test)
pred = xgbmodel.predict_proba(test)[:,1]


# --------------------- Writing results

pred_df['click'] = pred
file_name = "Predictions\\prediction_" + str(datetime.datetime.now().date()) + "_" +\
            str(datetime.datetime.now().strftime("%H:%M:%S")) + ".csv"
pred_df.to_csv(path_or_buf="Predictions\\prediction_4.csv", index=False)