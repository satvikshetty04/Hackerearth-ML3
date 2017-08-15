'''
Building over Pre_processing1.1
'''

import datetime
import numpy as np
import pandas as pd

# --------------------- Initializing variables

start_time = datetime.datetime.now()
print("Started at: " + str(start_time))

# --------------------- Loading datasets

train = pd.read_csv("Data\\train_pp2.csv", header=0)
test = pd.read_csv("Data\\test_pp2.csv", header=0)

# --------------------- Filling siteid nulls

train['siteid'].fillna(-999, inplace=True)
test['siteid'].fillna(-999, inplace=True)


# --------------------- Creating weekday, weekend and minute features

train['weekday'] = pd.to_datetime(train['datetime']).dt.weekday
train['is_weekend'] = 0
train.loc[train['weekday'].isin([5,6]), 'is_weekend'] = 1
train['minute'] = pd.to_datetime(train['datetime']).dt.minute

test['weekday'] = pd.to_datetime(test['datetime']).dt.weekday
test['is_weekend'] = 0
test.loc[test['weekday'].isin([5,6]), 'is_weekend'] = 1
test['minute'] = pd.to_datetime(test['datetime']).dt.minute


# --------------------- Adding aggregate features

site_offer_count = train.groupby(['siteid','offerid']).size().reset_index()
site_offer_count.columns = ['siteid','offerid','site_offer_count']

site_offer_count_test = test.groupby(['siteid','offerid']).size().reset_index()
site_offer_count_test.columns = ['siteid','offerid','site_offer_count']

site_cat_count = train.groupby(['siteid','category']).size().reset_index()
site_cat_count.columns = ['siteid','category','site_cat_count']

site_cat_count_test = test.groupby(['siteid','category']).size().reset_index()
site_cat_count_test.columns = ['siteid','category','site_cat_count']

site_mcht_count = train.groupby(['siteid','merchant']).size().reset_index()
site_mcht_count.columns = ['siteid','merchant','site_mcht_count']

site_mcht_count_test = test.groupby(['siteid','merchant']).size().reset_index()
site_mcht_count_test.columns = ['siteid','merchant','site_mcht_count']

cat_mcht_count = train.groupby(['category','merchant']).size().reset_index()
cat_mcht_count.columns = ['category','merchant','cat_mcht_count']

cat_mcht_count_test = test.groupby(['category','merchant']).size().reset_index()
cat_mcht_count_test.columns = ['category','merchant','cat_mcht_count']

cat_hr_count = train.groupby(['category','hour']).size().reset_index()
cat_hr_count.columns = ['category','hour','cat_hr_count']

cat_hr_count_test = test.groupby(['category','hour']).size().reset_index()
cat_hr_count_test.columns = ['category','hour','cat_hr_count']

site_wday_count = train.groupby(['siteid','weekday']).size().reset_index()
site_wday_count.columns = ['siteid','weekday','site_wday_count']

site_wday_count_test = test.groupby(['siteid','weekday']).size().reset_index()
site_wday_count_test.columns = ['siteid','weekday','site_wday_count']

brow_cat_count = train.groupby(['browserid','category']).size().reset_index()
brow_cat_count.columns = ['browserid','category','brow_cat_count']

brow_cat_count_test = test.groupby(['browserid','category']).size().reset_index()
brow_cat_count_test.columns = ['browserid','category','brow_cat_count']

agg_df = [site_offer_count, site_cat_count, site_mcht_count, cat_mcht_count, cat_hr_count, site_wday_count, brow_cat_count]
agg_df_test = [site_offer_count_test, site_cat_count_test, site_mcht_count_test, cat_mcht_count_test,
               cat_hr_count_test, site_wday_count_test, brow_cat_count_test]

for x in agg_df:
    train = train.merge(x)

for x in agg_df_test:
    test = test.merge(x)


# --------------------- Writing results

train.to_csv('Data\\train_pp3.csv', index=False)
test.to_csv('Data\\test_pp3.csv', index=False)

print("Overall time taken: " + str(datetime.datetime.now() - start_time))