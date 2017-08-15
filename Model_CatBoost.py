import datetime
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

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

# colormap = plt.cm.viridis
# plt.figure(figsize=(18,18))
# plt.title('Pearson Correlation of Features', y=1.05, size=15)
# sns.heatmap(train[cols_to_use].astype(float).corr(),linewidths=0.1,vmax=1.0, yticklabels="auto", cmap=colormap, linecolor='white', annot=True)
# plt.savefig("Correlation.png")

# --------------------- cat model
np.random.seed(1000)
rows = np.random.choice(train.index.values, 1000000)
sampled_train = train.loc[rows]

trainX = sampled_train[cols_to_use]
trainY = sampled_train['click']

X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.5)
model = CatBoostClassifier(depth=10, iterations=30, learning_rate=0.2, eval_metric='AUC', verbose=True, random_seed=1,
                           calc_feature_importance=True)

del sampled_train
del train
# del test
del trainX
del trainY
del rows

model.fit(X_train
          , y_train
          , cat_features=cat_cols
          , eval_set=(X_test, y_test)
          , use_best_model=True
          )


pred = model.predict_proba(test[cols_to_use])[:, 1]

sub = pd.DataFrame({'ID': test['ID'], 'click': pred})
sub.to_csv('Predictions\\pred_cat_21.csv', index=False, float_format="%.6f")

print("Overall time taken: " + str(datetime.datetime.now() - start_time))