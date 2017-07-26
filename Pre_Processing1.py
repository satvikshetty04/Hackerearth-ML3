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

train = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 3\\train.csv",
                    header=0)

test = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 3\\test.csv",
                    header=0)

          # GJ

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# --------------------- Creating day, hour features

train['day'] = pd.to_datetime(train['datetime']).dt.day
train['hour'] = pd.to_datetime(train['datetime']).dt.hour
test['day'] = pd.to_datetime(test['datetime']).dt.day
test['hour'] = pd.to_datetime(test['datetime']).dt.hour



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


unique_id = list(set(train[train['devid'].isnull() & train['browserid'].isnull()]['offerid']))
unique_id = sorted(unique_id)

print(train)
print(test)



train.isnull().sum(axis=0)/train.shape[0]
        
print([train[train['offerid']==i]['devid'].mode() for i in unique_id])

print("Time taken: "+ str(datetime.datetime.now() - start_time))

