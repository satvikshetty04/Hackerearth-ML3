'''
Building a Model using XGBoost
'''

import sys
import datetime
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --------------------- Initializing variables

start_time = datetime.datetime.now()
print("Started at: " + str(start_time))
pred_df = pd.DataFrame()

# --------------------- Loading datasets

train = pd.read_csv("Data\\train_pp3.csv", header=0)
test = pd.read_csv("Data\\test_pp3.csv", header=0)


# --------------------- Label Encoding

encode_cols = ['countrycode', 'browserid', 'devid']
for col in encode_cols:
    le = LabelEncoder()
    le.fit(list(train[col].values) + list(test[col].values))
    train[col] = le.transform(list(train[col]))
    test[col] = le.transform(list(test[col]))

train_mod = train[train["click"] == 1]
train = train.append(train_mod, ignore_index=True)
train = train.append(train_mod, ignore_index=True)

# --------------------- Dropping columns
pred_df['ID'] = test['ID']
cols_to_drop = ['ID', 'datetime', 'siteid', 'weekday', 'minute']
train.drop(cols_to_drop, axis=1, inplace=True)
test.drop(cols_to_drop, axis=1, inplace=True)

# --------------------- Splitting data

Y = train['click']
train.drop('click', axis=1, inplace=True)
X = train
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=125)


# --------------------- Creating model

xgbclassifier = XGBClassifier(n_estimators=100, nthread=-1, silent=False, seed=125, learning_rate=0.2, max_depth=7)
# xgbmodel = xgbclassifier.fit(X_train, y_train)
xgbmodel = xgbclassifier.fit(X, Y)

# pred = xgbmodel.predict(test)
# xgbmodel.score(X_test, y_test)
pred = xgbmodel.predict_proba(test)[:,1]


# --------------------- Writing results

pred_df['click'] = pred
file_name = "Predictions\\prediction_" + str(datetime.datetime.now().date()) + "_" +\
            str(datetime.datetime.now().strftime("%H%M%S")) + ".csv"
pred_df.to_csv(path_or_buf="Predictions\\prediction_35_depth7_rate2_iter100.csv", index=False)