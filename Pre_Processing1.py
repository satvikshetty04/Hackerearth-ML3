'''
Pre-Processing file.
'''

import sys
import datetime
import numpy as np
import pandas as pd
import os


# --------------------- Initializing variables

start_time = datetime.datetime.now()
print("Started at: " + str(start_time))


# --------------------- Loading dataset

# SS
train = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 3\\train.csv",
                    header=0)

test = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 3\\test.csv",
                    header=0)

# GJ
# train = pd.read_csv("train.csv")
# test = pd.read_csv("test.csv")


# --------------------- Creating day, hour features

train['day'] = pd.to_datetime(train['datetime']).dt.day
train['hour'] = pd.to_datetime(train['datetime']).dt.hour
test['day'] = pd.to_datetime(test['datetime']).dt.day
test['hour'] = pd.to_datetime(test['datetime']).dt.hour


# --------------------- Replacing null values - device ID

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

unique_offerid = list(set(train[train['devid'].isnull() & train['browserid'].isnull()]['offerid']))
unique_offerid = sorted(unique_offerid)

train.isnull().sum(axis=0)/train.shape[0]
        
for i in unique_offerid:
    print(train[train['offerid']==i]['devid'].mode())

print(train)
print(test)

print("Time taken: " + str(datetime.datetime.now() - start_time))

