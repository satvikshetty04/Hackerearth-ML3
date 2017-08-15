'''
Merging predictions
'''

import sys
import datetime
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from vecstack import stacking
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score


# --------------------- Initializing variables

start_time = datetime.datetime.now()
print("Started at: " + str(start_time))

# --------------------- Loading datasets

train = pd.read_csv("Data\\train_pp3.csv", header=0)
test = pd.read_csv("Data\\test_pp3.csv", header=0)


# --------------------- Oversampling 1's data

train_mod = train[train["click"] == 1]
train = train.append(train_mod, ignore_index=True)
train = train.append(train_mod, ignore_index=True)


cols = ['siteid', 'offerid', 'category', 'merchant', 'countrycode', 'browserid', 'devid', 'hour_range', 'is_weekend']

for x in cols:
    train[x] = train[x].astype('object')
    test[x] = test[x].astype('object')

cols_to_use = [x for x in train.columns if x not in ['ID', 'datetime', 'click', 'minute', 'brow_cat_count']]

encode_cols = ['countrycode', 'browserid', 'devid']
for col in encode_cols:
    le = LabelEncoder()
    le.fit(list(train[col].values) + list(test[col].values))
    train[col] = le.transform(list(train[col]))
    test[col] = le.transform(list(test[col]))

# catboost accepts categorical variables as indexes
cat_cols = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11]



# --------------------- cat model
np.random.seed(1000)
rows = np.random.choice(train.index.values, 1500000)
sampled_train = train.loc[rows]

trainX = sampled_train[cols_to_use]
trainY = sampled_train['click']

X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.3, random_state=0)

models = [CatBoostClassifier(depth=10, iterations=30, learning_rate=0.2, eval_metric='AUC', verbose=True, random_seed=1,
                           calc_feature_importance=True),
          CatBoostClassifier(depth=11, iterations=30, learning_rate=0.2, eval_metric='AUC', verbose=True, random_seed=1,
                             calc_feature_importance=True),
          CatBoostClassifier(depth=9, iterations=30, learning_rate=0.2, eval_metric='AUC', verbose=True, random_seed=1,
                             calc_feature_importance=True)
          ]

S_train, S_test = stacking(models, trainX.values, trainY.values, test[cols_to_use].values,
    regression = False, metric = accuracy_score, n_folds = 4, stratified = True, shuffle=True, random_state = 0, verbose = 2)

catclassifier = CatBoostClassifier(depth=10, iterations=20, learning_rate=0.2, eval_metric='AUC', verbose=True, random_seed=1,
                           calc_feature_importance=True)

# Fit 2-nd level model
model = catclassifier.fit(S_train, trainY)

# Predict
y_pred = model.predict_proba(S_test)[:,1]

pred_df = pd.DataFrame()
pred_df['ID'] = test['ID']
pred_df['click'] = y_pred
pred_df.to_csv(path_or_buf="Predictions\\stacked_2.csv", index=False)
